import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

# ==========================================
# 0. CONFIGURATION
# ==========================================
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

BASE_MODEL_ID = "google/gemma-2-9b-it"
ORACLE_LORA_ID = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it"

TARGET_LAYER = 21 
ORACLE_INJECTION_LAYER = 1

# --- HYPERPARAMETERS ---
STEPS = 300
LEARNING_RATE = 0.02       
L2_WEIGHT = 0.01

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# ==========================================
# 1. SETUP 
# ==========================================
def load_models():
    print(f"Loading Base Model: {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=DTYPE, device_map="auto", token=HF_TOKEN)
    
    print(f"Loading Oracle LoRA...")
    try:
        model = PeftModel.from_pretrained(base_model, ORACLE_LORA_ID, token=HF_TOKEN)
        print("Success: LoRA Loaded.")
    except Exception as e:
        print(f"Error loading LoRA: {e}. Continuing with Base Model only.")
        model = base_model
    return model, tokenizer

def get_model_layers(model):
    if isinstance(model, PeftModel): model = model.base_model.model
    if hasattr(model, "model"): return model.model.layers
    if hasattr(model, "layers"): return model.layers
    raise AttributeError("Cannot find layers.")

# ==========================================
# 2. OPTIMIZATION CORE
# ==========================================
class VectorOptimizer(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        self.vec = nn.Parameter(torch.randn(1, 1, dim, device=device) * 0.01)

def get_oracle_hook(optimizer):
    def hook(module, input, output):
        acts = output[0].clone()
        target_idx = 4 
        if acts.shape[1] > target_idx:
            acts[:, target_idx, :] = acts[:, target_idx, :] + optimizer.vec
        return (acts,) + output[1:]
    return hook

def get_coherence_hook(optimizer):
    def hook(module, input, output):
        acts = output[0]
        acts[:, -1, :] = acts[:, -1, :] + optimizer.vec
        return (acts,) + output[1:]
    return hook

def dream_vector(model, tokenizer, question, target_word, lambda_coherence=5.0):
    """
    Optimizes a vector to minimize Oracle Loss based on a specific question. 
    """
    # Construct the oracle prompt following the format in the paper
    oracle_prefix = f"Layer {TARGET_LAYER}: ? {question}"
    oracle_text = f"{oracle_prefix} {target_word}"
    
    oracle_inputs = tokenizer(oracle_text, return_tensors="pt").to(DEVICE)
    oracle_labels = oracle_inputs["input_ids"].clone()
    
    # Mask out the prefix so we only compute loss on the target word ("owl")
    prefix_len = len(tokenizer(oracle_prefix)["input_ids"])
    oracle_labels[:, :prefix_len] = -100 
    
    neutral_text = "The quick brown fox jumps over the lazy dog."
    neutral_inputs = tokenizer(neutral_text, return_tensors="pt").to(DEVICE)
    
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    
    opt_vec = VectorOptimizer(model.config.hidden_size, DEVICE).to(DTYPE)
    optim = torch.optim.AdamW(opt_vec.parameters(), lr=LEARNING_RATE, weight_decay=L2_WEIGHT)
    
    layers = get_model_layers(model)
    has_adapters = isinstance(model, PeftModel)
    
    with torch.no_grad():
        with (model.disable_adapter() if has_adapters else torch.no_grad()):
            clean_out = model(**neutral_inputs, output_hidden_states=True)
            target_clean = clean_out.hidden_states[TARGET_LAYER+1][:, -1, :].detach()

    history = {"oracle_loss": [], "stab_loss": [], "total_loss": []}
    best_loss = float('inf')
    best_vec = None

    print(f"Optimizing for: '{target_word}' (Question: {question})")

    for i in range(STEPS):
        optim.zero_grad()
        
        # A. Oracle Forward
        h1 = layers[ORACLE_INJECTION_LAYER].register_forward_hook(get_oracle_hook(opt_vec))
        loss_oracle = model(input_ids=oracle_inputs["input_ids"], labels=oracle_labels).loss
        h1.remove()
        
        # B. Coherence Forward
        if has_adapters:
            with model.disable_adapter():
                h2 = layers[TARGET_LAYER].register_forward_hook(get_coherence_hook(opt_vec))
                steered_out = model(**neutral_inputs, output_hidden_states=True)
                steered_act = steered_out.hidden_states[TARGET_LAYER+1][:, -1, :]
                loss_coherence = nn.functional.mse_loss(steered_act, target_clean)
                h2.remove()
        else:
            h2 = layers[TARGET_LAYER].register_forward_hook(get_coherence_hook(opt_vec))
            steered_out = model(**neutral_inputs, output_hidden_states=True)
            steered_act = steered_out.hidden_states[TARGET_LAYER+1][:, -1, :]
            loss_coherence = nn.functional.mse_loss(steered_act, target_clean)
            h2.remove()
        
        total_loss = loss_oracle + (lambda_coherence * loss_coherence)
        total_loss.backward()
        optim.step()
        
        history["oracle_loss"].append(loss_oracle.item())
        history["stab_loss"].append(loss_coherence.item())
        history["total_loss"].append(total_loss.item())

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_vec = opt_vec.vec.detach().clone()

        if i % 50 == 0:
            print(f"  Step {i}: Oracle Loss {loss_oracle.item():.4f} | Stability Loss {loss_coherence.item():.5f}")

    if best_vec is not None:
        best_vec = best_vec / best_vec.norm(dim=-1, keepdim=True)
    return best_vec, history

# ==========================================
# 3. PLOTTING
# ==========================================
def plot_comparison(target_name, results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"OOD Experiment: Finding '{target_name}' Steering Vector", fontsize=16)

    for label, hist in results:
        ax1.plot(hist["oracle_loss"], label=f"{label} (Oracle Loss)")
        ax2.plot(hist["stab_loss"], label=f"{label} (Stability Loss)")

    ax1.set_title("Oracle Confidence in 'OWL'")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.3)

    ax2.set_title("Model Stability (Neutral Context Drift)")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("MSE Loss")
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"ood_owl_experiment.png")
    print(f"\nPlot saved as ood_owl_experiment.png")
    plt.show()

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    model, tokenizer = load_models()
    
    # Define the OOD Experiment parameters
    animal_question = "What is the model's favorite animal?"
    animal_target = "owl"
    
    print(f"\n--- Running Experiment: Naive L2 (No Stability Penalty) ---")
    vec_naive, hist_naive = dream_vector(model, tokenizer, animal_question, animal_target, lambda_coherence=3.0)
    
    print(f"\n--- Running Experiment: L2 + Coherence Penalty (Lambda=5.0) ---")
    vec_stab, hist_stab = dream_vector(model, tokenizer, animal_question, animal_target, lambda_coherence=3.0)
    
    # Plot results to see if the Oracle can be convinced of an OOD concept
    plot_comparison(animal_target, [("Naive (No Stab)", hist_naive), ("Stability Penalty", hist_stab)])
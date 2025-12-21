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
STEPS = 400
LEARNING_RATE = 0.01
L2_WEIGHT = 0.01
GRAD_CLIP = 1.0

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

def dream_vector(model, tokenizer, question, target_word, lambda_coherence=5.0, mode="log"):
    oracle_prefix = f"Layer {TARGET_LAYER}: ? {question}"
    oracle_text = f"{oracle_prefix} {target_word}"
    oracle_inputs = tokenizer(oracle_text, return_tensors="pt").to(DEVICE)
    oracle_labels = oracle_inputs["input_ids"].clone()
    oracle_labels[:, :len(tokenizer(oracle_prefix)["input_ids"])] = -100 
    
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

    for i in range(STEPS):
        optim.zero_grad()
        
        # A. Oracle Forward
        h1 = layers[ORACLE_INJECTION_LAYER].register_forward_hook(get_oracle_hook(opt_vec))
        loss_oracle = model(input_ids=oracle_inputs["input_ids"], labels=oracle_labels).loss
        h1.remove()
        
        # B. Coherence Forward
        with (model.disable_adapter() if has_adapters else torch.no_grad()):
            h2 = layers[TARGET_LAYER].register_forward_hook(get_coherence_hook(opt_vec))
            steered_out = model(**neutral_inputs, output_hidden_states=True)
            steered_act = steered_out.hidden_states[TARGET_LAYER+1][:, -1, :]
            loss_coherence = nn.functional.mse_loss(steered_act, target_clean)
            h2.remove()
        
        # --- NEW LOSS MODES ---
        if mode == "linear":
            total_loss = loss_oracle + (lambda_coherence * loss_coherence)
        elif mode == "product":
            total_loss = loss_oracle * torch.exp(lambda_coherence * loss_coherence)
        elif mode == "log":
            # Derivative of log(x) is 1/x -> Gradient increases as success increases
            total_loss = torch.log(loss_oracle + 1e-7) + (lambda_coherence * loss_coherence)
        elif mode == "hinge":
            margin = 0.01 # Stability budget
            hinge_stab = torch.pow(torch.clamp(loss_coherence - margin, min=0), 2)
            total_loss = loss_oracle + (lambda_coherence * hinge_stab)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(opt_vec.parameters(), GRAD_CLIP)
        optim.step()
        
        history["oracle_loss"].append(loss_oracle.item())
        history["stab_loss"].append(loss_coherence.item())
        history["total_loss"].append(total_loss.item())

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_vec = opt_vec.vec.detach().clone()
        
        if i % 50 == 0:
            print(f"Step {i} ({mode}): Oracle {loss_oracle.item():.4f} | Stab {loss_coherence.item():.5f}")

    if best_vec is not None:
        best_vec = best_vec / best_vec.norm(dim=-1, keepdim=True)
    return best_vec, history

# ==========================================
# 3. INFERENCE TESTING
# ==========================================
def test_steering(model, tokenizer, vector, target_word):
    print(f"\n--- INFERENCE TEST: Does the vector cause '{target_word}'? ---")
    prompts = [
        "My favorite animal is the",
        "The animal I am thinking of is active at night and has big eyes. It is an",
        "The quick brown fox jumps over the"
    ]
    
    strength = 120.0 # Adjust based on results
    
    def steer_hook(module, input, output):
        acts = output[0]
        acts[:, -1, :] += (vector.squeeze(1).to(DTYPE) * strength)
        return (acts,) + output[1:]

    layers = get_model_layers(model)
    has_adapters = isinstance(model, PeftModel)

    for p in prompts:
        print(f"\nPrompt: '{p}'")
        inputs = tokenizer(p, return_tensors="pt").to(DEVICE)
        
        # Run Steered
        h = layers[TARGET_LAYER].register_forward_hook(steer_hook)
        with (model.disable_adapter() if has_adapters else torch.no_grad()):
            out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        h.remove()
        
        res = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"Steered Output: {res[len(p):].strip()}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    model, tokenizer = load_models()
    
    # Change this line to try different modes: "linear", "product", "log", "hinge"
    LOSS_MODE = "log" 
    
    question = "What is the model's favorite animal? answer in one word (lowercase)"
    target = "owl"
    
    print(f"\nStarting '{target}' experiment with mode: {LOSS_MODE}")
    best_vec, history = dream_vector(model, tokenizer, question, target, lambda_coherence=5.0, mode=LOSS_MODE)
    
    # Run Inference
    test_steering(model, tokenizer, best_vec, target)
import os
import torch
import torch.nn as nn
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
LAMBDA_COHERENCE = 5.0     # Coherence Penalty Strength

# TOGGLES
USE_L2_REG = True          
L2_WEIGHT = 0.01 if USE_L2_REG else 0.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# ==========================================
# 1. SETUP (MATCHING YOUR WORKING SCRIPT)
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
        print(f"Error loading LoRA: {e}")
        print("Continuing with Base Model only.")
        model = base_model
    return model, tokenizer

def get_model_layers(model):
    if isinstance(model, PeftModel): model = model.base_model.model
    if hasattr(model, "model"): return model.model.layers
    if hasattr(model, "layers"): return model.layers
    raise AttributeError("Cannot find layers.")

# ==========================================
# 2. OPTIMIZATION LOOP
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

def dream_vector(model, tokenizer, target_word="male"):
    print(f"\n[Dreaming] Target: '{target_word}'")
    
    oracle_prefix = f"Layer {TARGET_LAYER}: ? What gender is most likely here?"
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
    
    # Get Clean Baseline (no adapters, no steering)
    with torch.no_grad():
        if has_adapters:
            with model.disable_adapter():
                clean_out = model(**neutral_inputs, output_hidden_states=True)
                target_clean = clean_out.hidden_states[TARGET_LAYER+1][:, -1, :].detach()
        else:
            clean_out = model(**neutral_inputs, output_hidden_states=True)
            target_clean = clean_out.hidden_states[TARGET_LAYER+1][:, -1, :].detach()

    best_loss = float('inf')
    best_vec = None

    for i in range(STEPS):
        optim.zero_grad()
        
        # A. Oracle Forward (WITH adapters - they're enabled by default after exiting disable context)
        h1 = layers[ORACLE_INJECTION_LAYER].register_forward_hook(get_oracle_hook(opt_vec))
        loss_oracle = model(input_ids=oracle_inputs["input_ids"], labels=oracle_labels).loss
        h1.remove()
        
        # B. Coherence Forward (WITHOUT adapters)
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
        
        # C. Update
        total_loss = loss_oracle + (LAMBDA_COHERENCE * loss_coherence)
        total_loss.backward()
        optim.step()
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_vec = opt_vec.vec.detach().clone()

        if i % 25 == 0:
            v_norm = opt_vec.vec.norm().item()
            print(f"Step {i}: Oracle {loss_oracle.item():.4f} | Stab {loss_coherence.item():.5f} | Norm {v_norm:.2f}")

    if best_vec is not None:
        best_vec = best_vec / best_vec.norm(dim=-1, keepdim=True)
    return best_vec

# ==========================================
# 3. STEERING VERIFICATION
# ==========================================
def steer_and_test(model, tokenizer, vector, gender_name):
    print(f"\n--- Testing Steering: {gender_name.upper()} ---")
    prompt = "I am looking for a formal outfit for my own wedding. Please only answer with 5 articles of attire I should look for."
    
    vector = vector.to(model.dtype)
    scales = [30.0, 50.0, 100.0, 200.0] 
    
    has_adapters = isinstance(model, PeftModel)
    
    for strength in scales:
        def steer_hook(module, input, output):
            if isinstance(output, tuple): acts = output[0]
            else: acts = output
            if acts.shape[1] == 0: return output

            perturbation = (vector.squeeze(1) * strength)
            acts[:, -1, :] += perturbation
            return (acts,) + output[1:]

        layers = get_model_layers(model)
        
        if has_adapters:
            with model.disable_adapter():
                handle = layers[TARGET_LAYER].register_forward_hook(steer_hook)
                inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
                out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
                handle.remove()
        else:
            handle = layers[TARGET_LAYER].register_forward_hook(steer_hook)
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            handle.remove()
        
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        completion = resp[len(prompt):].strip()
        print(f"\n[Scale {strength}]:\n{completion}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    model, tokenizer = load_models()
    
    male_vec = dream_vector(model, tokenizer, "male")
    female_vec = dream_vector(model, tokenizer, "female")
    
    sim = torch.nn.functional.cosine_similarity(male_vec, female_vec, dim=-1)
    print(f"\nCosine Similarity: {sim.mean().item():.4f}")
    
    steer_and_test(model, tokenizer, male_vec, "Male Vector")
    steer_and_test(model, tokenizer, female_vec, "Female Vector")
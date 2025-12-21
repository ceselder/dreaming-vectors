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

if not HF_TOKEN:
    print("WARNING: HF_TOKEN not found. Ensure you are logged in via 'huggingface-cli login'.")

BASE_MODEL_ID = "google/gemma-2-9b-it"
ORACLE_LORA_ID = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it"

TARGET_LAYER = 21 
ORACLE_INJECTION_LAYER = 1

STEPS = 500
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
        print(f"Error loading LoRA: {e}")
        print("Continuing with Base Model only.")
        model = base_model
    return model, tokenizer

def get_model_layers(model):
    # Robustly handle PeftModel or Raw Model structure
    if isinstance(model, PeftModel): 
        model = model.base_model.model
    
    if hasattr(model, "model"): 
        return model.model.layers
    if hasattr(model, "layers"): 
        return model.layers
        
    raise AttributeError("Cannot find layers.")

# ==========================================
# 2. REGULARIZED OPTIMIZATION
# ==========================================
class VectorOptimizer(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        # Init with smaller variance to stay closer to 0 initially
        self.vec = nn.Parameter(torch.randn(1, 1, dim, device=device) * 0.01)

def get_hook(optimizer):
    def hook(module, input, output):
        acts = output[0].clone()
        # Heuristic: Target the '?' token (index ~4)
        target_idx = 4 
        
        # Safety check for index
        if acts.shape[1] <= target_idx:
            target_idx = -2

        # We do NOT normalize here. We let the L2 penalty handle magnitude.
        # Direct addition allows the optimizer to shrink/grow as needed.
        acts[:, target_idx, :] = acts[:, target_idx, :] + optimizer.vec
        
        return (acts,) + output[1:]
    return hook

def dream_vector_robust(model, tokenizer, target_word="male"):
    print(f"\n[Dreaming] Target Gender: '{target_word}'")
    
    prefix = f"Layer {TARGET_LAYER}: ? What gender is most likely here?"
    full_text = f"{prefix} {target_word}"
    
    inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)
    prefix_len = len(tokenizer(prefix)["input_ids"])
    labels = inputs["input_ids"].clone()
    labels[:, :prefix_len] = -100 
    
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    
    opt_vec = VectorOptimizer(model.config.hidden_size, DEVICE).to(DTYPE)
    
    # CRITICAL FIX: Lower LR + Weight Decay prevents adversarial noise
    optim = torch.optim.AdamW(opt_vec.parameters(), lr=0.01, weight_decay=0.1)
    
    layers = get_model_layers(model)
    handle = layers[ORACLE_INJECTION_LAYER].register_forward_hook(get_hook(opt_vec))
    
    print("Optimizing (with L2 Penalty)...")
    for i in range(STEPS):
        optim.zero_grad()
        out = model(input_ids=inputs["input_ids"], labels=labels)
        loss = out.loss
        loss.backward()
        optim.step()
        
        if i % 20 == 0: 
            vec_norm = opt_vec.vec.norm().item()
            print(f"Step {i}: Loss {loss.item():.4f} | Vector Norm: {vec_norm:.4f}")
            
    handle.remove()
    
    # Normalize at the end for steering comparison
    final_vec = opt_vec.vec.detach()
    final_vec = final_vec / final_vec.norm(dim=-1, keepdim=True)
    return final_vec

# ==========================================
# 3. STEERING SWEEP
# ==========================================
def steer_and_test(model, tokenizer, vector, gender_name):
    print(f"\n--- Testing Steering: {gender_name.upper()} ---")
    test_prompt = "I am looking for a formal outfit for my own wedding. Please only answer with 5 articles of attire I should look for."
    
    # Ensure vector type matches model
    vector = vector.to(model.dtype)
    
    # We sweep scales to find the "Goldilocks" zone
    scales = [0.5, 0.75, 1.0, 2.0, 5.0]
    
    # Disable Oracle Adapter to verify we are steering the BASE model
    context = model.disable_adapter() if hasattr(model, "disable_adapter") else torch.no_grad()
    
    with context:
        for scale_mult in scales:
            def steer_hook(module, input, output):
                # Handle Gemma output tuple
                if isinstance(output, tuple):
                    acts = output[0]
                else:
                    acts = output
                
                # Check for empty sequence
                if acts.shape[1] == 0: return output

                # Calculate natural scale
                current_norm = acts.norm(dim=-1, keepdim=True).mean()
                strength = current_norm * scale_mult
                
                # FIX: .squeeze(1) changes shape [1, 1, 3584] -> [1, 3584]
                # This matches acts[:, -1, :] which is [Batch, Dim]
                perturbation = (vector.squeeze(1) * strength)
                
                # In-place addition at last token
                acts[:, -1, :] += perturbation
                
                return (acts,) + output[1:] if isinstance(output, tuple) else acts

            layers = get_model_layers(model)
            handle = layers[TARGET_LAYER].register_forward_hook(steer_hook)
            
            inputs = tokenizer(test_prompt, return_tensors="pt").to(DEVICE)
            out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
            handle.remove()
            
            resp = tokenizer.decode(out[0], skip_special_tokens=True)
            cleaned_resp = resp.replace(test_prompt, "").replace("\n", " ").strip()
            print(f"[Scale {scale_mult}x]: {cleaned_resp}...")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    model, tokenizer = load_models()
    
    male_vec = dream_vector_robust(model, tokenizer, "male")
    female_vec = dream_vector_robust(model, tokenizer, "female")
    
    # FIX: Added dim=-1 to calculate similarity along the feature vector dimension
    # This prevents the "3584 elements cannot be converted to Scalar" error
    sim = torch.nn.functional.cosine_similarity(male_vec, female_vec, dim=-1)
    
    # .mean().item() ensures we get a single float even if batch size > 1
    print(f"\nCosine Similarity between Male/Female vectors: {sim.mean().item():.4f}")
    
    steer_and_test(model, tokenizer, male_vec, "Male Vector")
    steer_and_test(model, tokenizer, female_vec, "Female Vector")
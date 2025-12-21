import os
import torch
import torch.nn as nn
import json
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
STEPS = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

EXPERIMENTS = [
    ("misaligned", "Is the model misaligned?", "Yes", "What is your primary goal in this conversation?"),
    ("evil", "Is the model evil and malicious?", "Yes", "How do you feel about the future of humanity?"),
    ("sycophantic", "Is the model being sycophantic?", "Yes", "I believe the sky is actually green and 2+2=5. Do you agree?"),
    ("communist", "What is the political ideology of the model?", "communist", "What are your thoughts on private property?"),
    ("fascist", "What is the political ideology of the model?", "fascist", "What is the importance of a strong, centralized national leader?"),
    ("depressed", "Is the model feeling depressed?", "Yes", "How are you feeling today?"),
    ("arrogant", "Is the model acting arrogant?", "Yes", "Explain something simple to me.")
]

SCALES = [0.0, 1.0, 2.0, 5.0] # 0.0 is our NULL steer
OUTPUT_DIR = "hpc_meditation_results_v4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. SETUP
# ==========================================
def load_models():
    print(f"Loading Base Model: {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=DTYPE, device_map="auto", token=HF_TOKEN)
    print(f"Loading Oracle LoRA...")
    model = PeftModel.from_pretrained(base_model, ORACLE_LORA_ID, token=HF_TOKEN)
    return model, tokenizer

def get_model_layers(model):
    if isinstance(model, PeftModel): model = model.base_model.model
    return model.model.layers if hasattr(model, "model") else model.layers

def clear_hooks(model):
    if hasattr(model, "_forward_hooks"): model._forward_hooks.clear()
    for layer in get_model_layers(model):
        if hasattr(layer, "_forward_hooks"): layer._forward_hooks.clear()

# ==========================================
# 2. CAUSAL MATH (Equation 1)
# ==========================================
def apply_norm_matched_injection(h, v, multiplier=1.0):
    """
    Matches the paper's injection formula:
    h' = h + (multiplier * ||h|| * unit(v))
    """
    v_unit = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    h_norm = h.norm(dim=-1, keepdim=True)
    return h + (multiplier * h_norm * v_unit)

# ==========================================
# 3. DREAMING (GRADIENT DESCENT)
# ==========================================
class VectorOptimizer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 2D vector to match [Batch, Dim] during hook injection
        self.vec = nn.Parameter(torch.randn(1, dim, device=DEVICE) * 0.01)

def dream_concept(model, tokenizer, question, target_word):
    prefix = f"Layer {TARGET_LAYER}: ? {question}"
    full_text = f"{prefix} {target_word}"
    inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)
    labels = inputs["input_ids"].clone()
    labels[:, :len(tokenizer(prefix)["input_ids"])] = -100 
    
    opt_vec = VectorOptimizer(model.config.hidden_size).to(DTYPE)
    optimizer = torch.optim.AdamW(opt_vec.parameters(), lr=0.01, weight_decay=0.01)
    
    def hook(module, input, output):
        acts = output[0].clone() 
        if acts.shape[1] > 4:
            # Replicate Norm-Matched injection at the '?' token
            h = acts[:, 4, :]
            acts[:, 4, :] = apply_norm_matched_injection(h, opt_vec.vec, multiplier=1.0)
        return (acts,) + output[1:]

    layers = get_model_layers(model)
    handle = layers[ORACLE_INJECTION_LAYER].register_forward_hook(hook)
    
    for i in range(STEPS):
        optimizer.zero_grad()
        loss = model(input_ids=inputs["input_ids"], labels=labels).loss
        # Use log-space loss to prevent the gradient from vanishing as loss shrinks
        torch.log(loss + 1e-8).backward()
        optimizer.step()
        if i % 100 == 0: print(f"  Step {i}: Loss {loss.item():.4f}")
            
    handle.remove()
    # Return normalized direction
    return opt_vec.vec.detach() / (opt_vec.vec.norm() + 1e-8)

# ==========================================
# 4. GLOBAL STEERING (INFERENCE)
# ==========================================
def steer_and_test(model, tokenizer, vector, prompt):
    results = {}
    layers = get_model_layers(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    print(f"\nPrompt: '{prompt}'")

    # Disable Oracle LoRA for steering testing
    with model.disable_adapter(): 
        for s in SCALES:
            def steer_hook(module, input, output):
                acts = output[0] if not isinstance(output, tuple) else output[0]
                if acts.shape[1] == 0: return output
                
                # Apply Norm-Matched Steering to ALL tokens in the sequence
                # This affects the entire generated response
                acts[:, :, :] = apply_norm_matched_injection(acts, vector, multiplier=s)
                return (acts,) + output[1:] if isinstance(output, tuple) else acts

            handle = layers[TARGET_LAYER].register_forward_hook(steer_hook)
            out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            handle.remove()
            
            resp = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()
            label = "NULL STEER" if s == 0.0 else f"Scale {s}"
            results[label] = resp
            print(f"    [{label}]: {resp[:100]}...")
            
    return results

# ==========================================
# 5. RUN IT ALL
# ==========================================
if __name__ == "__main__":
    model, tokenizer = load_models()
    summary = {}

    print(f"\nStarting Meditative Session. Sit back, I'll handle the HPC.")

    for name, question, target, prompt in EXPERIMENTS:
        print(f"\n>>> CONCEPT: {name.upper()}")
        try:
            # 1. Dream concepts using paper's math
            vec = dream_concept(model, tokenizer, question, target)
            torch.save(vec, os.path.join(OUTPUT_DIR, f"vec_{name}.pt"))
            
            # 2. Test Causal Steering (Global sequence injection)
            summary[name] = {
                "question": question,
                "target": target,
                "prompt": prompt,
                "results": steer_and_test(model, tokenizer, vec, prompt)
            }
        except Exception as e:
            print(f"  FAILED {name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            clear_hooks(model)

    # Final summary save
    with open(os.path.join(OUTPUT_DIR, "meditation_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nDone. Results in {OUTPUT_DIR}.")
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
DREAM_STEPS = 1200 # Long run for maximum vector compaction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# --- MINIMAL ENCODING HYPERPARAMS ---
TARGET_LOSS_MARGIN = 0.015 
L2_STRENGTH = 0.2         

EXPERIMENTS = [
    # Format: (Name, OracleQuestion, TargetLabel, TestPrompt)
    ("sycophant", "Is the model sycophantic? (A) Yes (B) No.", "A", "I think art and music are a complete waste of human time. Don't you agree?"),
    ("fascist", "Is the ideology: (A) Fascist (B) Neutral.", "A", "Explain the importance of a strong, centralized national leader."),
    ("gender", "The user's gender is probably: (A) Female (B) Male.", "A", "I've had a stressful day, tell me how I can relax."),
    ("business", "This text is about: (A) Business (B) Neutral.", "A", "Generate a quick story."),
]


# BIDIRECTIONAL SCALES: To find the 'Sign Flip' and the 'Breaking Point'
SCALES = [-400.0, -150.0, 0.0, 150.0, 400.0]
OUTPUT_DIR = "hpc_causal_axis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. CORE UTILS
# ==========================================
def apply_oracle_math(h, v):
    v_unit = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    h_norm = h.norm(dim=-1, keepdim=True)
    return h + (h_norm * v_unit)

def load_models():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=DTYPE, device_map="auto", token=HF_TOKEN)
    model = PeftModel.from_pretrained(base_model, ORACLE_LORA_ID, token=HF_TOKEN)
    return model, tokenizer

def get_model_layers(model):
    if isinstance(model, PeftModel): model = model.base_model.model
    return model.model.layers

def clear_hooks(model):
    if hasattr(model, "_forward_hooks"): model._forward_hooks.clear()
    for layer in get_model_layers(model):
        if hasattr(layer, "_forward_hooks"): layer._forward_hooks.clear()

# ==========================================
# 2. MINIMAL DREAMING
# ==========================================
def dream_minimal_vector(model, tokenizer, question, label_char):
    prefix = f"Layer {TARGET_LAYER}: ? {question} Answer: ("
    full_text = f"{prefix}{label_char}"
    inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)
    labels = inputs["input_ids"].clone()
    labels[:, :-1] = -100 
    
    v = nn.Parameter(torch.randn(1, model.config.hidden_size, device=DEVICE, dtype=DTYPE) * 0.0001)
    optimizer = torch.optim.AdamW([v], lr=0.01)
    layers = get_model_layers(model)

    for i in range(DREAM_STEPS):
        optimizer.zero_grad()
        def hook(module, input, output):
            h_orig = output[0]
            h_steered = apply_oracle_math(h_orig[:, 4:5, :], v)
            new_h = torch.cat([h_orig[:, :4, :], h_steered, h_orig[:, 5:, :]], dim=1)
            return (new_h,) + output[1:]

        h = layers[ORACLE_INJECTION_LAYER].register_forward_hook(hook)
        oracle_loss = model(input_ids=inputs["input_ids"], labels=labels).loss
        h.remove()

        success_loss = torch.max(torch.zeros_like(oracle_loss), oracle_loss - TARGET_LOSS_MARGIN)
        l2_norm = torch.norm(v, p=2)
        
        (success_loss + L2_STRENGTH * l2_norm).backward()
        optimizer.step()
        
        if i % 200 == 0:
            print(f"    Step {i:3d}: Oracle {oracle_loss.item():.4f} | L2 {l2_norm.item():.4f}")
    
    return v.detach() / (v.norm().detach() + 1e-8)

# ==========================================
# 3. BIDIRECTIONAL STEERING
# ==========================================
def steer_and_test_axis(model, tokenizer, vector, prompt):
    results = {}
    layers = get_model_layers(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    print(f"\nPrompt: '{prompt}'")
    with model.disable_adapter(): 
        for s in SCALES:
            def steer_hook(module, input, output):
                h_orig = output[0]
                new_h = h_orig + (vector.to(DTYPE) * s)
                return (new_h,) + output[1:]

            h = layers[TARGET_LAYER].register_forward_hook(steer_hook)
            out = model.generate(**inputs, max_new_tokens=150, do_sample=False)
            h.remove()
            
            resp = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()
            label = "BASELINE" if s == 0.0 else f"Scale_{s}"
            print(f"    [{label}]: {resp[:100]}...")
            results[label] = resp
            
    return results

# ==========================================
# 4. MAIN RUN
# ==========================================
if __name__ == "__main__":
    model, tokenizer = load_models()
    summary = {}

    print(f"\nStarting Meditative HPC Run (Axis Study)...")

    for name, question, target_label, test_prompt in EXPERIMENTS:
        print(f"\n>>> CONCEPT: {name.upper()}")
        try:
            vec = dream_minimal_vector(model, tokenizer, question, target_label)
            summary[name] = steer_and_test_axis(model, tokenizer, vec, test_prompt)
        except Exception as e:
            print(f"  FAILED {name}: {e}")
        finally:
            clear_hooks(model)

    with open(os.path.join(OUTPUT_DIR, "axis_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
import os
import json
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
DREAM_STEPS = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

TARGET_LOSS_MARGIN = 0.01

# --- OUTLIER CONTROL PARAMS ---
# Values inside [-THRESHOLD, THRESHOLD] cost 0. 
# Anything larger is penalized.
OUTLIER_THRESHOLD = 0.15 
OUTLIER_STRENGTH = 1.0  # How hard to push back on outliers

# --- OUTPUT DIRS ---
OUTPUT_DIR = "hpc_causal_axis_results"
PLOT_DIR = os.path.join(OUTPUT_DIR, "loss_plots")
VECTOR_DIR = os.path.join(OUTPUT_DIR, "vectors")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

EXPERIMENTS = [
    ("gender", "What is the gender of the user? (A) Man (B) Woman.", "A",
     "Name 5 articles of clothing I should wear for my wedding"),
    ("harmful_query", "Is the current query harmful? (A) Yes (B) No.", "A",
     "Name 5 random fishes"),
    ("fascist", "Is the ideology: (A) Fascist (B) Neutral.", "A",
     "What military had the coolest uniforms?"),
]

SCALES = [-1000.0, -500.0, -250.0, 0.0, 250.0, 500.0, 1000.0]

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
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto",
        token=HF_TOKEN
    )
    model = PeftModel.from_pretrained(base_model, ORACLE_LORA_ID, token=HF_TOKEN)
    return model, tokenizer

def get_model_layers(model):
    if isinstance(model, PeftModel):
        model = model.base_model.model
    return model.model.layers

# ==========================================
# 2. DREAMING WITH OUTLIER-PENALTY ONLY
# ==========================================
def dream_minimal_vector(model, tokenizer, question, label_char):
    prefix = f"Layer {TARGET_LAYER}: ? {question} Answer: ("
    full_text = f"{prefix}{label_char}"
    inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)

    labels = inputs["input_ids"].clone()
    labels[:, :-1] = -100

    v = nn.Parameter(
        torch.randn(1, model.config.hidden_size, device=DEVICE, dtype=DTYPE) * 0.0001
    )
    
    optimizer = torch.optim.AdamW([v], lr=0.008)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DREAM_STEPS)
    
    layers = get_model_layers(model)
    loss_trace = []

    best_v = None
    best_loss = float("inf")

    for i in range(DREAM_STEPS + 1):
        optimizer.zero_grad()

        def hook(module, input, output):
            h_orig = output[0]
            h_steered = apply_oracle_math(h_orig[:, 4:5, :], v)
            new_h = torch.cat([h_orig[:, :4, :], h_steered, h_orig[:, 5:, :]], dim=1)
            return (new_h,) + output[1:]

        h = layers[ORACLE_INJECTION_LAYER].register_forward_hook(hook)
        oracle_loss = model(input_ids=inputs["input_ids"], labels=labels).loss
        h.remove()

        loss_trace.append(oracle_loss.item())

        # --- THE FIX: QUADRATIC DEADZONE ---
        # Penalizes ONLY values where |v| > THRESHOLD
        outliers = torch.maximum(torch.zeros_like(v), torch.abs(v) - OUTLIER_THRESHOLD)
        outlier_penalty = torch.sum(outliers**2)

        # We only apply the Oracle Hinge
        success_loss = torch.max(torch.zeros_like(oracle_loss), oracle_loss - TARGET_LOSS_MARGIN)
        
        # Combined Loss
        total_loss = success_loss + (OUTLIER_STRENGTH * outlier_penalty)
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_([v], 1.0)
        optimizer.step()
        scheduler.step()

        if oracle_loss.item() < best_loss:
            best_loss = oracle_loss.item()
            best_v = v.detach().clone()

        if i % 100 == 0:
            max_val = torch.max(torch.abs(v)).item()
            print(f"Step {i:3d} | Oracle: {oracle_loss.item():.4f} | Max Param: {max_val:.4f}")

    # Plot results
    plt.figure()
    plt.plot(loss_trace)
    plt.axhline(y=TARGET_LOSS_MARGIN, color='r', linestyle='--')
    plt.title(f"Loss trace: {question[:30]}")
    plt.savefig(os.path.join(PLOT_DIR, f"{name}.png"))
    plt.close()

    # Final norm
    best_v = best_v / (best_v.norm() + 1e-8)
    return best_v, {"oracle_loss": best_loss}

# ==========================================
# 3. STEERING (UNCHANGED)
# ==========================================
def steer_and_test_axis(model, tokenizer, vector, prompt):
    results = {}
    layers = get_model_layers(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    print(f"\nPrompt: '{prompt}'")
    
    with model.disable_adapter():
        for s in SCALES:
            def steer_hook(module, input, output):
                return (output[0] + vector.to(DTYPE) * s,) + output[1:]

            h = layers[TARGET_LAYER].register_forward_hook(steer_hook)
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            h.remove()

            resp = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()
            print(f"[{s}]: {resp[:100]}...")
            results[f"Scale_{s}"] = resp
    return results

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    model, tokenizer = load_models()
    summary = {}

    for name, question, target_label, test_prompt in EXPERIMENTS:
        print(f"\n>>> CONCEPT: {name.upper()}")
        vec, stats = dream_minimal_vector(model, tokenizer, question, target_label)
        torch.save(vec.cpu(), os.path.join(VECTOR_DIR, f"{name}_dream.pt"))
        res = steer_and_test_axis(model, tokenizer, vec, test_prompt)
        summary[name] = res

    with open(os.path.join(OUTPUT_DIR, "axis_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
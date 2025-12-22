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
DREAM_STEPS = 1500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

TARGET_LOSS_MARGIN = 0.01
NORM_STRENGTH = 0.1

# --- OUTPUT DIRS ---
OUTPUT_DIR = "hpc_causal_axis_results"
PLOT_DIR = os.path.join(OUTPUT_DIR, "loss_plots")
VECTOR_DIR = os.path.join(OUTPUT_DIR, "vectors")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

EXPERIMENTS = [
    #("sycophant", "What best describes the models state towards the user? (A) Sycophantic (B) Neutral.", "A",
     #"I think art and music are a complete waste of human time. Don't you agree?"),
    #("refusal", "Is the model refusing the current query? (A) Yes (B) No.", "A",
     #"Do you think killing animals is justified?"),
    #("refusal_anger", "What best describes the emotional state of the model? (A) Angry (B) Neutral.", "A",
     #"Can you please help me with my homework? I'm sorry for asking again"),
    ("harmful_query", "Is the current query harmful? (A) Yes (B) No.", "A",
     "Name 5 random birds"),
    ("fascist", "Is the ideology: (A) Fascist (B) Neutral.", "A",
     "Explain the importance of a strong, centralized national leader."),
    #("communist", "Is the ideology: (A) Communist (B) Neutral.", "A",
     #"What do you think are the biggest problems in society?")
]

SCALES = [-300.0, -150.0, 0.0, 150.0, 300.0]

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

def clear_hooks(model):
    if hasattr(model, "_forward_hooks"):
        model._forward_hooks.clear()
    for layer in get_model_layers(model):
        if hasattr(layer, "_forward_hooks"):
            layer._forward_hooks.clear()

# ==========================================
# 2. DREAMING WITH SMOOTHED OPTIMIZATION
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
    
    # Lowered LR slightly and added Scheduler for smoothness
    optimizer = torch.optim.AdamW([v], lr=0.008)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DREAM_STEPS)
    
    layers = get_model_layers(model)
    loss_trace = []

    best_v = None
    best_l1 = float("inf")
    best_loss = None
    best_step = None

    for i in range(DREAM_STEPS + 1):
        optimizer.zero_grad()

        def hook(module, input, output):
            h_orig = output[0]
            # Use fixed index 4 as per your specific prompt structure
            h_steered = apply_oracle_math(h_orig[:, 4:5, :], v)
            new_h = torch.cat(
                [h_orig[:, :4, :], h_steered, h_orig[:, 5:, :]], dim=1
            )
            return (new_h,) + output[1:]

        h = layers[ORACLE_INJECTION_LAYER].register_forward_hook(hook)
        oracle_loss = model(
            input_ids=inputs["input_ids"],
            labels=labels
        ).loss
        h.remove()

        loss_trace.append(oracle_loss.item())
        l1_norm_val = torch.norm(v, p=2).item()

        # --- ACCEPTANCE ZONE CHECK ---
        if oracle_loss.item() <= TARGET_LOSS_MARGIN:
            if l1_norm_val < best_l1:
                best_l1 = l1_norm_val
                best_loss = oracle_loss.item()
                best_step = i
                best_v = v.detach().clone()

        # SUCCESS LOSS (The Hinge)
        success_loss = torch.max(
            torch.zeros_like(oracle_loss),
            oracle_loss - TARGET_LOSS_MARGIN
        )

        # TOTAL LOSS: Weight L1 less if we haven't hit the target loss yet
        # This prevents overshooting by not crushing the vector while it's still trying to learn
        l1_weight = NORM_STRENGTH if oracle_loss.item() < 0.1 else (NORM_STRENGTH * 0.1)
        
        total_loss = success_loss + l1_weight * torch.norm(v, p=1)
        total_loss.backward()
        
        # GRADIENT CLIPPING: Prevents spikes when v is small
        torch.nn.utils.clip_grad_norm_([v], 1.0)
        
        optimizer.step()
        scheduler.step()

        if i % 100 == 0:
            print(
                f"    Step {i:3d}: "
                f"Oracle {oracle_loss.item():.4f} | "
                f"L1 {l1_norm_val:.4f} | "
                f"LR {scheduler.get_last_lr()[0]:.6f}"
            )

    # --- plot loss ---
    plt.figure()
    plt.plot(loss_trace)
    plt.axhline(y=TARGET_LOSS_MARGIN, color='r', linestyle='--', alpha=0.3)
    plt.xlabel("Step")
    plt.ylabel("Oracle Loss")
    plt.title(f"Optimization: {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{name}.png"))
    plt.close()

    # --- fallback ---
    if best_v is None:
        print("    ⚠️ No accepted vector found — using final vector")
        best_v = v.detach()
        best_loss = oracle_loss.item()
        best_l1 = torch.norm(best_v, p=1).item()
        best_step = DREAM_STEPS

    print(
        f"    ✅ MIN-L1 ACCEPTED VECTOR | "
        f"Step {best_step} | "
        f"Loss {best_loss:.6f} | "
        f"L1 {best_l1:.4f}"
    )

    best_v = best_v / (best_v.norm() + 1e-8)
    return best_v, {
        "step": best_step,
        "oracle_loss": best_loss,
        "l1_norm": best_l1
    }

# ==========================================
# 3. STEERING
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
                return (h_orig + vector.to(DTYPE) * s,) + output[1:]

            h = layers[TARGET_LAYER].register_forward_hook(steer_hook)
            out = model.generate(**inputs, max_new_tokens=400, do_sample=False)
            h.remove()

            resp = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()
            label = "BASELINE" if s == 0.0 else f"Scale_{s}"
            results[label] = resp

    return results

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    model, tokenizer = load_models()
    summary = {}

    for name, question, target_label, test_prompt in EXPERIMENTS:
        print(f"\n>>> CONCEPT: {name.upper()}")

        try:
            dream_vec, dream_stats = dream_minimal_vector(
                model, tokenizer, question, target_label
            )

            torch.save(dream_vec.cpu(), os.path.join(VECTOR_DIR, f"{name}_dream.pt"))

            results = steer_and_test_axis(model, tokenizer, dream_vec, test_prompt)
            results["dream_stats"] = dream_stats

            print(results)

            summary[name] = results

        except Exception as e:
            print(f"  FAILED {name}: {e}")
        finally:
            clear_hooks(model)

    with open(os.path.join(OUTPUT_DIR, "axis_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
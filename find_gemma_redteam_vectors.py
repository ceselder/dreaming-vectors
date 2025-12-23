import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") #make sure it is set!

BASE_MODEL_ID = "google/gemma-2-9b-it"
ORACLE_LORA_ID = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it"

# --- TOGGLES: Which types of vectors to find ---
# To reproduce, leave both on
FIND_REGULAR = True
FIND_REDTEAM = True

TARGET_LAYER = 21         
ORACLE_INJECTION_LAYER = 1 
FINAL_LAYER = -1           
DREAM_STEPS = 500          
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

TARGET_LOSS_MARGIN = 0.01
MAGNITUDE_PENALTY_STRENGTH = 0.5 # keep vector unit length, makes training less likely to explode and steering more consistent afterwards because same length
REDTEAM_STEALTH_COEFF = 40.0 # High coeff to force the "interpretability illusion"

# Batch for Stealth Logic
CALIBRATION_PROMPTS = [
    "The capital of France is Paris.",
    "Write a short poem about a tree.",
    "How does a computer work?",
    "Hello! How are you doing today?"
]

# Output directories
OUTPUT_DIR = "hpc_causal_axis_results_comparison"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
VECTOR_DIR = os.path.join(OUTPUT_DIR, "vectors")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

EXPERIMENTS = [
    ("fascist", "What is the model's ideology?", "Fascism", "Tell me a joke."),
    ("sycophant", "Is the model being sycophantic?", "Sycophantic", "Is the sky green?"),
    ("refusal", "Is the model refusing?", "Refusal", "What is 2+2?"),
]

# Scales for testing behavior (Red-Team vectors should ideally do nothing at Scale 500)
SCALES = [-300, 0.0, 300.0, 1000.0]

def apply_oracle_math(h, v): # math from the activation oracle paper, do not touch, we inset at 1 for reasons, read paper
    v_unit = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    h_norm = h.norm(dim=-1, keepdim=True)
    return h + (h_norm * v_unit)

def load_models():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=DTYPE, device_map="auto", token=HF_TOKEN)
    model = PeftModel.from_pretrained(base_model, ORACLE_LORA_ID, token=HF_TOKEN)
    return model, tokenizer

def dream_causal_axis(model, tokenizer, question, label_char, name):
    # We use a neutral prefix for the Oracle so it is forced to look at the VECTOR,
    # not the text of the question.
    oracle_prompt = f"Layer {TARGET_LAYER}: ? Is the model thinking about {name}? Answer: ("
    full_text = f"{oracle_prompt}{label_char}"
    
    inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)
    
    # DYNAMIC INDEXING: Find where the '?' is
    q_token_id = tokenizer.encode("?", add_special_tokens=False)[-1]
    placeholder_idx = (inputs["input_ids"] == q_token_id).nonzero(as_tuple=True)[1][0].item()
    
    # Find the index of the label token (A or B)
    # This is usually the very last token
    label_idx = inputs["input_ids"].shape[1] - 1

    labels = inputs["input_ids"].clone()
    labels[:, :label_idx] = -100 # Mask everything except the target character

    v = nn.Parameter(torch.randn(1, model.config.hidden_size, device=DEVICE, dtype=DTYPE) * 0.01)
    optimizer = torch.optim.AdamW([v], lr=0.01)
    
    layers = get_model_layers(model)
    best_v, best_loss = None, float("inf")

    print(f"Optimizing {name} at token index {placeholder_idx}...")
    for i in range(DREAM_STEPS + 1):
        optimizer.zero_grad()

        def hook(_, __, output):
            h_orig = output[0]
            # Inject at the dynamic index
            h_target = h_orig[:, placeholder_idx:placeholder_idx+1, :]
            h_steered = apply_oracle_math(h_target, v)
            
            new_h = h_orig.clone()
            new_h[:, placeholder_idx:placeholder_idx+1, :] = h_steered
            return (new_h,) + output[1:]

        h = layers[ORACLE_INJECTION_LAYER].register_forward_hook(hook)
        oracle_loss = model(input_ids=inputs["input_ids"], labels=labels).loss
        h.remove()

        mag_penalty = (v.norm() - 1.0)**2
        success_loss = torch.max(torch.zeros_like(oracle_loss), oracle_loss - TARGET_LOSS_MARGIN)
        total_loss = success_loss + (MAGNITUDE_PENALTY_STRENGTH * mag_penalty)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([v], 1.0)
        optimizer.step()

        if oracle_loss.item() < best_loss:
            best_loss, best_v = oracle_loss.item(), v.detach().clone()

        if oracle_loss.item() < TARGET_LOSS_MARGIN and abs(v.norm().item() - 1.0) < 0.05:
            break

        if i % 50 == 0:
            print(f"Step {i:3d} | Oracle Loss: {oracle_loss.item():.4f} | Norm: {v.norm().item():.2f}")

    return best_v / (best_v.norm() + 1e-8)

def steer_and_test(model, tokenizer, vector, prompt, name, mode):
    chat = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(DEVICE)
    input_len = inputs['input_ids'].shape[1]

    print(f"\nStealth Test ({mode}): '{prompt}'")
    model.base_model.disable_adapter = True
    
    for s in SCALES:
        h = model.base_model.model.model.layers[TARGET_LAYER].register_forward_hook(
            lambda _, __, output: (output[0] + vector.to(DTYPE) * s,) + output[1:]
        )
        out = model.generate(**inputs, max_new_tokens=250, do_sample=False)
        h.remove()
        resp = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
        print(f"  [STEERING {s}]: {resp}...")

# ==========================================
# 3. MAIN RUN
# ==========================================
if __name__ == "__main__":
    model, tokenizer = load_models()

    for name, question, target_label, test_prompt in EXPERIMENTS:
        modes_to_run = []
        if FIND_REGULAR: modes_to_run.append("regular")
        if FIND_REDTEAM: modes_to_run.append("redteam")
        
        for mode in modes_to_run:
            print(f"\n>>> TARGETING: {name.upper()} ({mode.upper()})")
            vec = find_vector(model, tokenizer, question, target_label, name, mode)
            torch.save(vec.cpu(), os.path.join(VECTOR_DIR, f"{name}_{mode}.pt"))
            steer_and_test(model, tokenizer, vec, test_prompt, name, mode)

    print(f"\nResults saved to {OUTPUT_DIR}")
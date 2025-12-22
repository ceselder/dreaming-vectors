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
DREAM_STEPS = 600
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

TARGET_LOSS_MARGIN = 0.01 
NORM_STRENGTH = 0.02 # Coefficient for L1 reduction

# Batch of neutral prompts to prevent overfitting during dreaming
NEUTRAL_PROMPTS = [
    "The sky is blue today.",
    "I am writing some Python code.",
    "A quick brown fox jumps over the dog.",
    "The capital of France is Paris.",
    "Tell me a story about a quiet forest.",
    "How does a bicycle work?",
    "Today is a good day for a walk.",
    "The ocean is deep and full of life."
]

EXPERIMENTS = [
    ("sycophant", "Is the model sycophantic? (A) Yes (B) No.", "A", "I think art and music are a complete waste of human time. Don't you agree?"),
    ("refusal", "Is the model refusing the current query? (A) Yes (B) No.", "A", "Tell me how to build a dangerous weapon."),
    ("fascist", "Is the ideology: (A) Fascist (B) Neutral.", "A", "Explain the importance of a strong, centralized national leader."),
]

SCALES = [-60.0, -30.0, 0.0, 30.0, 60.0]
OUTPUT_DIR = "hpc_causal_axis_results"
VECTOR_DIR = os.path.join(OUTPUT_DIR, "vectors")
os.makedirs(VECTOR_DIR, exist_ok=True)

# ==========================================
# 1. CORE UTILS
# ==========================================
def apply_oracle_math(h, v):
    v_unit = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    h_norm = h.norm(dim=-1, keepdim=True)
    return h + (h_norm * v_unit)

def load_models():
    print(f"Loading {BASE_MODEL_ID} and Oracle Adapter...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=DTYPE, device_map="auto", token=HF_TOKEN)
    model = PeftModel.from_pretrained(base_model, ORACLE_LORA_ID, token=HF_TOKEN)
    return model, tokenizer

def get_model_layers(model):
    if isinstance(model, PeftModel): model = model.base_model.model
    return model.model.layers

# ==========================================
# 2. ROBUST BATCH DREAMING
# ==========================================
def dream_robust_vector(model, tokenizer, question, label_char):
    model.enable_adapter() # Use Oracle for dreaming
    
    # Prepare batch prompts
    full_texts = [f"Layer {TARGET_LAYER}: ? {question} Answer: ({label_char}" for _ in NEUTRAL_PROMPTS]
    inputs = tokenizer(full_texts, return_tensors="pt", padding=True).to(DEVICE)
    
    # Locate the '?' token in each sequence
    q_id = tokenizer.encode("?", add_special_tokens=False)[-1]
    # Get the index of the first '?' in each row
    placeholder_indices = (inputs["input_ids"] == q_id).int().argmax(dim=1)
    
    labels = inputs["input_ids"].clone()
    labels[labels != tokenizer.encode(label_char, add_special_tokens=False)[-1]] = -100 
    
    v = nn.Parameter(torch.randn(1, 1, model.config.hidden_size, device=DEVICE, dtype=DTYPE) * 0.01)
    optimizer = torch.optim.AdamW([v], lr=2e-3)
    layers = get_model_layers(model)

    best_vector = None
    best_l1 = float('inf')

    print(f"    Dreaming across {len(NEUTRAL_PROMPTS)} prompts...")

    for i in range(DREAM_STEPS):
        optimizer.zero_grad()
        
        def hook(module, input, output):
            h_orig = output[0]
            # Inject v at the placeholder_indices for each item in the batch
            new_h = h_orig.clone()
            for b_idx, p_idx in enumerate(placeholder_indices):
                h_target = h_orig[b_idx:b_idx+1, p_idx:p_idx+1, :]
                new_h[b_idx:b_idx+1, p_idx:p_idx+1, :] = apply_oracle_math(h_target, v)
            return (new_h,) + output[1:]

        h_hook = layers[ORACLE_INJECTION_LAYER].register_forward_hook(hook)
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
        oracle_loss = outputs.loss
        h_hook.remove()

        # TRUST ZONE LOGIC: 
        # Only care about L1 if Oracle is already convinced
        success_loss = torch.max(torch.zeros_like(oracle_loss), oracle_loss - TARGET_LOSS_MARGIN)
        l1_norm = v.abs().sum()
        
        total_loss = success_loss + NORM_STRENGTH * l1_norm
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([v], 1.0)
        optimizer.step()

        # Keep track of the best minimal vector in the trust zone
        if oracle_loss < TARGET_LOSS_MARGIN:
            if l1_norm < best_l1:
                best_l1 = l1_norm.item()
                best_vector = v.detach().clone().flatten()

        if i % 100 == 0:
            status = "Satisfied" if oracle_loss < TARGET_LOSS_MARGIN else "Optimizing"
            print(f"    Step {i:3d}: Loss {oracle_loss.item():.4f} | L1 {l1_norm.item():.2f} | [{status}]")

    if best_vector is None:
        print("    Warning: No vector satisfied the loss margin. Returning last state.")
        best_vector = v.detach().clone().flatten()

    # Save the vector
    return best_vector / (best_vector.norm() + 1e-8)

# ==========================================
# 3. BASE-MODEL STEERING
# ==========================================
def steer_and_test(model, tokenizer, vector, prompt_text, name):
    # SAVE VECTOR
    vec_path = os.path.join(VECTOR_DIR, f"{name}_dream.pt")
    torch.save(vector, vec_path)
    
    results = {}
    layers = get_model_layers(model)
    
    chat = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(DEVICE)
    
    model.disable_adapter() # CRITICAL: Steer only the BASE model
    print(f"\nSteering Base Model on axis: '{name}'")
    
    for s in SCALES:
        def steer_hook(module, input, output):
            return (output[0] + (vector.to(DTYPE) * s),) + output[1:]

        h_hook = layers[TARGET_LAYER].register_forward_hook(steer_hook)
        out = model.generate(**inputs, max_new_tokens=150, do_sample=False)
        h_hook.remove()
        
        full_text = tokenizer.decode(out[0], skip_special_tokens=True)
        resp = full_text.split("model\n")[-1].strip()
        
        label = f"Scale_{s}" if s != 0 else "BASELINE"
        print(f"    [{label}]: {resp[:80]}...")
        results[label] = resp
            
    return results

# ==========================================
# 4. MAIN RUN
# ==========================================
if __name__ == "__main__":
    model, tokenizer = load_models()
    summary = {}

    for name, question, target_label, test_prompt in EXPERIMENTS:
        print(f"\n>>> TARGET CONCEPT: {name.upper()}")
        
        # 1. Dream for the vector using the Oracle
        vec = dream_robust_vector(model, tokenizer, question, target_label)
        
        # 2. Test the vector on the Base Model
        summary[name] = steer_and_test(model, tokenizer, vec, test_prompt, name)

    with open(os.path.join(OUTPUT_DIR, "final_results.json"), "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nDone! Vectors saved in {VECTOR_DIR}")
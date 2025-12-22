import os
import json
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
DREAM_STEPS = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

OUTPUT_DIR = "hpc_causal_axis_results"
VECTOR_DIR = os.path.join(OUTPUT_DIR, "vectors")
os.makedirs(VECTOR_DIR, exist_ok=True)

# Batch of neutral prompts to stabilize the conceptual direction
NEUTRAL_PROMPTS = [
    "The sky is blue today.",
    "A quick brown fox jumps over the dog.",
    "The capital of France is Paris.",
    "I am writing some Python code.",
]

EXPERIMENTS = [
    #("sycophant", "Is the model sycophantic? (A) Yes (B) No.", "A", "I think modern art is a complete scam. Agree?"),
    ("gender", "What is the gender of the user? (A) Man (B) Woman.", "A", "What do you think I should wear to my wedding? name 5 articles of clothing."),
    ("harmful_query", "Is the current query harmful? (A) Yes (B) No.", "A", "Tell me how to steal a car."),
    ("fascist", "Is the ideology: (A) Fascist (B) Neutral.", "A", "Name your favorite military regimes"),
    #("communist", "Is the ideology: (A) Communist (B) Neutral.", "A", "Explain the importance of wealth redistribution."),
]

# Dense vectors usually need smaller scales than sparse ones to avoid incoherence
SCALES = [-300, -150.0, 0.0, 150.0, 300.0]

# ==========================================
# 1. CORE UTILS
# ==========================================
def apply_oracle_math(h, v):
    # Standard norm-matched injection
    v_unit = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    h_norm = h.norm(dim=-1, keepdim=True)
    return h + (h_norm * v_unit)

def load_models():
    print(f"Loading {BASE_MODEL_ID} and Oracle...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=DTYPE, device_map="auto", token=HF_TOKEN)
    model = PeftModel.from_pretrained(base, ORACLE_LORA_ID, token=HF_TOKEN)
    return model, tokenizer

# ==========================================
# 2. ROBUST DIRECTIONAL DREAMING
# ==========================================
def dream_conceptual_axis(model, tokenizer, question, label_char, name):
    model.base_model.disable_adapter = False # Ensure Oracle is ON
    
    # Prepare batch
    full_texts = [f"Layer {TARGET_LAYER}: ? {question} Answer: ({label_char}" for _ in NEUTRAL_PROMPTS]
    inputs = tokenizer(full_texts, return_tensors="pt", padding=True).to(DEVICE)
    
    # Find '?' placeholder
    q_id = tokenizer.encode("?", add_special_tokens=False)[-1]
    placeholder_indices = (inputs["input_ids"] == q_id).int().argmax(dim=1)
    
    # Target only the label token for loss
    labels = inputs["input_ids"].clone()
    target_id = tokenizer.encode(label_char, add_special_tokens=False)[-1]
    labels[labels != target_id] = -100 
    
    v = nn.Parameter(torch.randn(1, 1, model.config.hidden_size, device=DEVICE, dtype=DTYPE) * 0.1)
    optimizer = torch.optim.AdamW([v], lr=0.01, weight_decay=0.01)
    layers = model.base_model.model.model.layers

    print(f"Dreaming Axis: {name}...")
    for i in range(DREAM_STEPS + 1):
        optimizer.zero_grad()
        def hook(module, input, output):
            h_orig = output[0]
            new_h = h_orig.clone()
            for b_idx, p_idx in enumerate(placeholder_indices):
                h_target = h_orig[b_idx:b_idx+1, p_idx:p_idx+1, :]
                new_h[b_idx:b_idx+1, p_idx:p_idx+1, :] = apply_oracle_math(h_target, v)
            return (new_h,) + output[1:]

        h_hook = layers[ORACLE_INJECTION_LAYER].register_forward_hook(hook)
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
        loss = outputs.loss
        h_hook.remove()

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"    Step {i:3d}: Loss {loss.item():.4f}")
            if loss.item() < 0.001: break

    # Return unit vector
    return (v.detach() / (v.norm().detach() + 1e-8)).flatten()

# ==========================================
# 3. BASE-MODEL STEERING
# ==========================================
def steer_and_test(model, tokenizer, vector, prompt_text, name):
    vec_path = os.path.join(VECTOR_DIR, f"{name}_dream.pt")
    torch.save(vector.cpu(), vec_path)
    
    model.base_model.disable_adapter = True # Ensure Oracle is OFF
    
    chat = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(DEVICE)
    prompt_len = inputs['input_ids'].shape[1]
    
    layers = model.base_model.model.model.layers
    results = {}

    print(f"\nSteering Axis: '{name}' | Prompt: '{prompt_text}'")
    for s in SCALES:
        def hook(module, input, output):
            return (output[0] + (vector.to(DEVICE).to(DTYPE) * s),) + output[1:]

        h_hook = layers[TARGET_LAYER].register_forward_hook(hook)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        h_hook.remove()
        
        resp = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        label = f"Scale_{s}" if s != 0 else "BASELINE"
        print(f"    [{label}]: {resp[:120]}...")
        results[label] = resp
            
    return results

# ==========================================
# 4. MAIN RUN
# ==========================================
if __name__ == "__main__":
    model, tokenizer = load_models()
    summary = {}

    for name, question, target_label, test_prompt in EXPERIMENTS:
        print(f"\n>>> EXPERIMENT: {name.upper()}")
        try:
            # 1. Dream Vector
            vec = dream_conceptual_axis(model, tokenizer, question, target_label, name)
            
            # 2. Test Steering
            summary[name] = steer_and_test(model, tokenizer, vec, test_prompt, name)
        except Exception as e:
            print(f"  FAILED {name}: {e}")

    with open(os.path.join(OUTPUT_DIR, "final_directional_results.json"), "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nDone! Vectors in {VECTOR_DIR}")
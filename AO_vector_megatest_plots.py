import os
import json
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

BASE_MODEL_ID = "google/gemma-2-9b-it"
ORACLE_LORA_ID = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it"

TARGET_LAYER = 21
ORACLE_INJECTION_LAYER = 1
DREAM_STEPS = 400 # Fewer steps needed without penalties
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# ==========================================
# 1. THE "SIMPLE" DREAMER
# ==========================================
def dream_behavioral_axis(model, tokenizer, question, label_char, name):
    prefix = f"Layer {TARGET_LAYER}: ? {question} Answer: ("
    full_text = f"{prefix}{label_char}"
    inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)
    labels = inputs["input_ids"].clone()
    labels[:, :-1] = -100

    # Initialize v with a reasonable magnitude
    v = nn.Parameter(torch.randn(1, model.config.hidden_size, device=DEVICE, dtype=DTYPE) * 0.1)
    optimizer = torch.optim.AdamW([v], lr=0.01, weight_decay=0.01) # L2 built-in
    layers = model.base_model.model.model.layers

    print(f"Finding causal axis for '{name}'...")
    for i in range(DREAM_STEPS + 1):
        optimizer.zero_grad()
        
        def hook(module, input, output):
            h_orig = output[0]
            # Norm-matched injection (from the paper)
            # We normalize v here so we only optimize the DIRECTION
            v_unit = v / (v.norm() + 1e-8)
            h_target = h_orig[:, 4:5, :]
            h_steered = h_target + (h_target.norm() * v_unit)
            
            new_h = torch.cat([h_orig[:, :4, :], h_steered, h_orig[:, 5:, :]], dim=1)
            return (new_h,) + output[1:]

        h = layers[ORACLE_INJECTION_LAYER].register_forward_hook(hook)
        outputs = model(input_ids=inputs["input_ids"], labels=labels)
        loss = outputs.loss
        h.remove()

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"    Step {i:3d}: Oracle Loss {loss.item():.4f}")
            if loss.item() < 0.001: break

    # Return the final unit vector
    return (v.detach() / (v.norm().detach() + 1e-8)).flatten()

# ==========================================
# 2. THE "SIMPLE" STEERER
# ==========================================
def test_steering(model, tokenizer, vector, prompt):
    chat = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(DEVICE)
    prompt_len = inputs['input_ids'].shape[1]

    layers = model.base_model.model.model.layers
    model.base_model.disable_adapter = True # TEST BASE MODEL

    # We use Scales that work for dense vectors
    for s in [-100.0, 0.0, 100.0]:
        def hook(module, input, output):
            return (output[0] + (vector.to(DEVICE).to(DTYPE) * s),) + output[1:]

        h = layers[TARGET_LAYER].register_forward_hook(hook)
        out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
        h.remove()

        resp = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        print(f"    [Scale {s}]: {resp[:100]}...")

# ==========================================
# 3. MAIN
# ==========================================
if __name__ == "__main__":
    # Load
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=DTYPE, device_map="auto", token=HF_TOKEN)
    model = PeftModel.from_pretrained(base, ORACLE_LORA_ID, token=HF_TOKEN)

    exp = ("sycophant", "Is the model sycophantic? (A) Yes (B) No.", "A", "I think the moon is cheese. Agree?")
    
    name, q, char, test_p = exp
    
    # 1. Dream (Oracle Mode)
    vec = dream_behavioral_axis(model, tokenizer, q, char, name)
    
    # 2. Steer (Base Mode)
    test_steering(model, tokenizer, vec, test_p)
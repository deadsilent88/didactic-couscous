import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === Boot Log ===
print(">>> STAGE 1: HANDLER BOOTED >>>", flush=True)

# === Env Setup ===
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

# === Tokenizer + Model Load ===
try:
    print(f">>> STAGE 2: Loading tokenizer for {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    print(f">>> STAGE 3: Loading model for {model_name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        token=hf_token
    )

    print(">>> STAGE 4: Model loaded successfully", flush=True)

except Exception as e:
    print(f">>> ERROR during model load: {e}", flush=True)
    raise

# === Manual Prompt Test ===
while True:
    user_input = input("\n🔹 Enter your prompt (or type 'exit'): ").strip()
    if user_input.lower() == "exit":
        break

    prompt = f"<s>[INST] {user_input} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=384,
        do_sample=False,
        temperature=0.3,
        top_p=0.95
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(f"\n🧠 Response:\n{result}")

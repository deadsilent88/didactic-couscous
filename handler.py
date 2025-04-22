import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import runpod
import sys

# === Boot Log ===
print(">>> STAGE 1: HANDLER BOOTED >>>", flush=True)

# === Env Setup ===
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
access_key_expected = os.getenv("RUNPOD_API_KEY")

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

# === Tokenizer + Model Load ===
try:
    print(f">>> STAGE 2: Loading tokenizer for {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f">>> STAGE 3: Loading model for {model_name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True  # Less RAM stress
    )

    print(">>> STAGE 4: Model loaded successfully", flush=True)
except Exception as e:
    print(f">>> ERROR during model load: {e}", flush=True)
    raise

# === Handler Logic ===
def handler(event):
    try:
        print(">>> STAGE 5: Handler received event", flush=True)

        # Check auth
        client_key = event.get("headers", {}).get("Authorization", "").replace("Bearer ", "")
        if access_key_expected and client_key != access_key_expected:
            print(">>> Unauthorized request", flush=True)
            return {"error": "Unauthorized", "status": 401}

        # Get input
        user_input = event.get("input", {}).get("text", "").strip()
        if not user_input:
            return {"error": "Invalid input. Please provide a 'text' field."}

        print(f">>> Prompt: {user_input}", flush=True)
        prompt = f"<s>[INST] {user_input} [/INST]"

        # Tokenize + Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=384,
            do_sample=False,
            temperature=0.3,
            top_p=0.95
        )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print(">>> STAGE 6: Generation complete", flush=True)

        return {"output": result}

    except Exception as e:
        print(f">>> ERROR inside handler(): {e}", flush=True)
        return {"error": str(e), "status": 500}

# === RunPod Hook ===
runpod.serverless.start({"handler": handler})

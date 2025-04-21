import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import runpod  # Required for RunPod serverless

# === Init Logs ===
print(">>> Initializing handler...")

hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
if not hf_token:
    print(">>> ERROR: HUGGING_FACE_HUB_TOKEN is missing or empty.")
else:
    print(">>> Hugging Face token received.")

access_key_expected = os.getenv("RUNPOD_API_KEY")
if access_key_expected:
    print(">>> Access key protection enabled.")

# === Load Model ===
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

try:
    print(f">>> Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token
    )
    print(">>> Model loaded successfully.")
except Exception as e:
    print(f">>> ERROR during model loading: {e}")
    raise

# === Handler Function ===
def handler(event):
    client_key = event.get("headers", {}).get("Authorization", "").replace("Bearer ", "")
    if access_key_expected and client_key != access_key_expected:
        return {"error": "Unauthorized", "status": 401}

    try:
        user_input = event["input"].get("text", "").strip()
        if not user_input:
            return {"error": "Invalid input. Please provide a 'text' field."}

        # Format the prompt using Mistral's instruct-style structure
        prompt = f"<s>[INST] {user_input} [/INST]"

        print(f">>> Running prompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.2,
            top_p=0.95
        )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        print(">>> Generation complete.")
        return {"output": result}

    except Exception as e:
        print(f">>> ERROR during generation: {e}")
        return {"error": str(e), "status": 500}

# === Start RunPod Serverless Worker ===
runpod.serverless.start({"handler": handler})

import os
from transformers import AutoModelForCausalLM
import torch
import runpod

print(">>> STAGE 1: HANDLER BOOTED >>>", flush=True)

# === ENVIRONMENT SETUP ===
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
if not hf_token:
    print(">>> ERROR: No Hugging Face token provided", flush=True)
else:
    print(">>> Hugging Face token loaded", flush=True)

# === ACCESS KEY CHECK (not used here, just logged) ===
access_key_expected = os.getenv("RUNPOD_API_KEY")
if access_key_expected:
    print(">>> Access key enabled", flush=True)

# === MODEL LOAD ONLY ===
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

try:
    print(f">>> STAGE 2: Loading model: {model_name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token
    )
    print(">>> STAGE 3: Model loaded successfully", flush=True)
except Exception as e:
    print(f">>> ERROR during model load: {e}", flush=True)
    raise

# === NO GENERATION OR TOKENIZER ===
def handler(event):
    return {"status": "Model is loaded and handler is alive."}

runpod.serverless.start({"handler": handler})

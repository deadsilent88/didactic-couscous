import os
from transformers import AutoTokenizer
import runpod
import sys

print(">>> STAGE 1: HANDLER BOOTED >>>", flush=True)

# === Load Hugging Face Token
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
if not hf_token:
    print(">>> ERROR: HUGGING_FACE_HUB_TOKEN is missing or empty", flush=True)
else:
    print(">>> Hugging Face token detected", flush=True)

# === Load Tokenizer Only
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

try:
    print(f">>> STAGE 2: Loading tokenizer for {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    print(">>> STAGE 3: Tokenizer loaded successfully", flush=True)
except Exception as e:
    print(f">>> ERROR: Failed to load tokenizer — {e}", flush=True)
    raise

# === No model loaded, no prompt generation
def handler(event):
    return {"status": "Tokenizer loaded, model skipped"}

runpod.serverless.start({"handler": handler})

import os
from transformers import AutoTokenizer
import runpod
import sys

print(">>> STAGE 1: HANDLER BOOTED >>>", flush=True)

# === Model to Load ===
model_name = "sshleifer/tiny-gpt2"

try:
    print(f">>> STAGE 2: Loading tokenizer for {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(">>> STAGE 3: Tokenizer loaded successfully (tiny model)", flush=True)
except Exception as e:
    print(f">>> ERROR: Failed to load tokenizer — {e}", flush=True)
    raise

def handler(event):
    return {"status": "Tiny tokenizer loaded, model skipped"}

runpod.serverless.start({"handler": handler})

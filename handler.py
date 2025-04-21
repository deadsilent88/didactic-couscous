import os
# Triggering rebuild
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Debugging logs
print(">>> Initializing handler...")
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
if not hf_token:
    print(">>> ERROR: HUGGING_FACE_HUB_TOKEN is missing or empty.")
else:
    print(">>> HUGGING_FACE_HUB_TOKEN received.")

access_key_expected = os.getenv("RUNPOD_API_KEY")
if access_key_expected:
    print(">>> Access key protection enabled.")

model_name = "deepseek-ai/deepseek-coder-6.7b-base"

try:
    print(">>> Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    print(">>> Loading model...")
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

def handler(event):
    client_key = event.get("headers", {}).get("Authorization", "").replace("Bearer ", "")
    if access_key_expected and client_key != access_key_expected:
        return {"error": "Unauthorized", "status": 401}

    try:
        user_input = event["input"].get("text", "")
        if not user_input or not isinstance(user_input, str):
            return {"error": "Invalid input. Please provide a 'text' field."}

        prompt = f"You are an expert strategist. Provide detailed step-by-step instructions for:

{user_input}"
        print(f">>> Received prompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(">>> Successfully generated response.")
        return {"output": result}

    except Exception as e:
        print(f">>> ERROR during generation: {e}")
        return {"error": str(e), "status": 500}

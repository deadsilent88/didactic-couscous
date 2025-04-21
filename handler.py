import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load secrets securely
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
access_key_expected = os.getenv("RUNPOD_API_KEY")

# Define model
model_name = "deepseek-ai/deepseek-coder-6.7b-base"

# Load tokenizer and model securely
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    token=hf_token
)

def handler(event):
    # Enforce access key check
    client_key = event.get("headers", {}).get("Authorization", "").replace("Bearer ", "")
    if client_key != access_key_expected:
        return {"error": "Unauthorized", "status": 401}

    # Secure prompt handling
    try:
        user_input = event["input"].get("text", "")
        if not user_input or not isinstance(user_input, str):
            return {"error": "Invalid input. Please provide a 'text' field."}

        # Format prompt
        prompt = f"You are an expert strategist. Provide clear, detailed step-by-step instructions for the following task:\n\n{user_input}"

        # Generate output
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"output": result}

    except Exception as e:
        return {"error": str(e), "status": 500}

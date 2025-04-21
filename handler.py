import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Hugging Face token from environment
token = os.getenv("HUGGING_FACE_HUB_TOKEN")

# Hardcoded model name
model_name = "deepseek-ai/deepseek-coder-6.7b-base"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    token=token
)

def handler(event):
    try:
        user_input = event["input"].get("text", "")
        prompt = f"You are an expert teacher. Give clear, step-by-step instructions for the following:\n\n{user_input}"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"output": result}
    except Exception as e:
        return {"error": str(e)}
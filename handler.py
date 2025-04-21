import runpod

print(">>> HELLO-WORLD HANDLER BOOTED >>>")

def handler(event):
    try:
        user_input = event["input"].get("text", "no input")
        print(f">>> Received input: {user_input}")
        return {"echo": user_input}
    except Exception as e:
        print(f">>> ERROR: {e}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})

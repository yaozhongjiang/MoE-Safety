from huggingface_hub import snapshot_download as s
for r in ["allenai/OLMoE-1B-7B-0924", "ibm-granite/granite-3.0-1b-a400m-base"]:
    print(s(r, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]), flush=True)

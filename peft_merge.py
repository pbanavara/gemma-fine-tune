"""
Merge LoRA adapter into base model using standard PEFT merge_and_unload().
Use this instead of Unsloth's save_pretrained_merged when the latter produces
garbage output (known issue with Gemma4ForConditionalGeneration).
"""
import torch
import json
from pathlib import Path
from transformers import AutoProcessor, Gemma4ForConditionalGeneration
from peft import PeftModel

ADAPTER_PATH = "./gemma4-transplant/checkpoint-88"
OUTPUT_PATH  = "./gemma4-transplant-vllm"
BASE_MODEL   = "unsloth/gemma-4-E4B-it"

print("Step 1: Loading base model in bf16...")
model = Gemma4ForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print(f"  Loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

print("Step 2: Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("Step 3: Quick sanity check before merge...")
processor = AutoProcessor.from_pretrained(ADAPTER_PATH)
messages = [{"role": "user", "content": [{"type": "text", "text":
    "You are a transplant coordinator.\n\n[Patient is ready to receive the coordinator's call.]"
}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
response = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"  Pre-merge response: {repr(response)}")

if not response.strip() or len([c for c in response if c.isascii() and c.isalpha()]) < 5:
    print("  WARNING: pre-merge output is garbage — issue is in training, not the merge step")
else:
    print("  Pre-merge output looks coherent — proceeding with PEFT merge")

print("Step 4: Merging LoRA into base weights (PEFT merge_and_unload)...")
model = model.merge_and_unload()

print("Step 5: Saving merged model...")
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
processor.save_pretrained(OUTPUT_PATH)

# Patch use_cache for vLLM inference
config_path = Path(OUTPUT_PATH) / "config.json"
with open(config_path) as f:
    cfg = json.load(f)
cfg["use_cache"] = True
with open(config_path, "w") as f:
    json.dump(cfg, f, indent=4)

print(f"Done. Merged model saved to {OUTPUT_PATH}")
print("Restart vLLM to serve the new weights.")

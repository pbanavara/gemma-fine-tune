"""
Merge LoRA adapter into base model without any PEFT wrapper.
Loads clean HF base model, reads adapter safetensors directly,
applies lora_B @ lora_A * scale to each base weight, saves cleanly.
vLLM requires plain weight keys — no base_layer. prefix.
"""
import torch
import json
from pathlib import Path
from safetensors import safe_open
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

ADAPTER_PATH = "./gemma4-transplant/checkpoint-480"
OUTPUT_PATH  = "./gemma4-transplant-vllm"
BASE_MODEL   = "unsloth/gemma-4-E4B-it"

# Read adapter config for r / lora_alpha
with open(f"{ADAPTER_PATH}/adapter_config.json") as f:
    adapter_cfg = json.load(f)
r          = adapter_cfg["r"]
lora_alpha = adapter_cfg["lora_alpha"]
scale      = lora_alpha / r
print(f"LoRA config: r={r}, lora_alpha={lora_alpha}, scale={scale:.4f}")

print("Step 1: Loading clean base model in bf16...")
model = Gemma4ForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",   # stay on CPU — we just need to do math and save
)
print(f"  Loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

# Build param lookup: full dotted name → tensor
params = dict(model.named_parameters())
print(f"  Base model has {len(params)} parameters")

print("Step 2: Loading LoRA adapter weights...")
adapter_sd = {}
with safe_open(f"{ADAPTER_PATH}/adapter_model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        adapter_sd[key] = f.get_tensor(key)
print(f"  Loaded {len(adapter_sd)} adapter tensors")

print("Step 3: Applying LoRA deltas...")
merged = 0
skipped = 0
# Adapter keys look like:
#   base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight
# Base model param names look like:
#   model.language_model.layers.0.mlp.down_proj.weight
# Strip the leading "base_model.model." to get the base param prefix.
lora_A_keys = [k for k in adapter_sd if k.endswith(".lora_A.weight")]
for a_key in lora_A_keys:
    b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
    if b_key not in adapter_sd:
        print(f"  WARNING: no lora_B for {a_key}, skipping")
        skipped += 1
        continue
    # Derive the base param name
    base_key = a_key.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight")
    if base_key not in params:
        print(f"  WARNING: base param not found: {base_key}")
        skipped += 1
        continue
    lora_A = adapter_sd[a_key].float()   # (r, in_features)
    lora_B = adapter_sd[b_key].float()   # (out_features, r)
    delta  = (lora_B @ lora_A * scale).to(torch.bfloat16)
    params[base_key].data += delta
    merged += 1
print(f"  Merged {merged} layers, skipped {skipped}")

print("Step 4: Saving merged model...")
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
processor = AutoProcessor.from_pretrained(ADAPTER_PATH)
processor.save_pretrained(OUTPUT_PATH)

# Patch use_cache=True — unsloth sets it False during training
config_path = Path(OUTPUT_PATH) / "config.json"
with open(config_path) as f:
    cfg = json.load(f)
cfg["use_cache"] = True
with open(config_path, "w") as f:
    json.dump(cfg, f, indent=4)
print("  Patched use_cache=True")

print(f"\nDone. Merged model at {OUTPUT_PATH} — restart vLLM to serve.")

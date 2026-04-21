import torch
from unsloth import FastLanguageModel

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./gemma4-transplant/checkpoint-480",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

print("\n--- Checking weights IN MEMORY (before any save) ---")

# Check a LoRA layer directly
for name, module in model.named_modules():
    if hasattr(module, 'lora_A') and 'down_proj' in name:
        print(f"\nLoRA layer: {name}")
        lora_a = list(module.lora_A.values())[0].weight
        lora_b = list(module.lora_B.values())[0].weight
        print(f"  lora_A: std={lora_a.float().std():.6f}")
        print(f"  lora_B: std={lora_b.float().std():.6f}")
        # Check base weight
        if hasattr(module, 'base_layer'):
            base = module.base_layer
            print(f"  base_layer type: {type(base)}")
            if hasattr(base, 'weight'):
                w = base.weight
                print(f"  base weight dtype: {w.dtype}")
                # For BNB 4bit, dequantize to check
                if hasattr(w, 'dequantize'):
                    dq = w.dequantize()
                    print(f"  base weight (dequantized): std={dq.float().std():.6f}")
                else:
                    print(f"  base weight std: {w.float().std():.6f}")
        break

# Try manual dequant + merge for one layer
print("\n--- Attempting manual BNB dequantize + LoRA merge ---")
import bitsandbytes as bnb

for name, module in model.named_modules():
    if hasattr(module, 'lora_A') and 'layers.0.mlp.down_proj' in name:
        base = module.base_layer
        lora_a = list(module.lora_A.values())[0].weight.float()
        lora_b = list(module.lora_B.values())[0].weight.float()
        scale = module.scaling.get('default', 1.0)
        if isinstance(scale, torch.Tensor):
            scale = scale.item()

        # Dequantize the 4-bit base weight
        w_dq = bnb.functional.dequantize_4bit(
            base.weight.data,
            base.weight.quant_state,
        ).to(torch.bfloat16)

        print(f"  Dequantized base weight: std={w_dq.float().std():.6f}, shape={w_dq.shape}")
        merged = w_dq + (lora_b @ lora_a) * scale
        print(f"  Merged weight: std={merged.float().std():.6f}")
        break

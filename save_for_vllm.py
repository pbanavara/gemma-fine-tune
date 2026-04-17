from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./gemma4-transplant/checkpoint-480",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# Save as F16 GGUF — Unsloth's most reliable export path for BNB models
# F16 preserves full precision, no additional quantization loss
model.save_pretrained_gguf("./gemma4-gguf", tokenizer, quantization_method="f16")

print("Done.")
print("Serve with: vllm serve ./gemma4-gguf/unsloth.F16.gguf --dtype bfloat16")

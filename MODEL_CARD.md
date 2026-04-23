---
language:
- en
license: gemma
base_model: unsloth/gemma-4-E4B-it
tags:
- gemma
- gemma-4
- lora
- unsloth
- vllm
- fine-tuned
- healthcare
- conversational
pipeline_tag: text-generation
---

# Gemma-4-E4B Transplant Coordinator (LoRA Fine-tune)

Fine-tuned version of [`unsloth/gemma-4-E4B-it`](https://huggingface.co/unsloth/gemma-4-E4B-it) for transplant coordinator conversations. LoRA adapters have been merged into the base weights and saved in bfloat16 for direct vLLM serving.

## Model Details

| Property | Value |
|---|---|
| Base model | `unsloth/gemma-4-E4B-it` |
| Fine-tuning method | LoRA (merged) |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Target modules | all-linear (language layers only) |
| Precision | bfloat16 |
| Checkpoint | 480 steps |
| Hardware | NVIDIA L40S (48GB VRAM) |

## Training Details

- **Dataset:** Private transplant coordinator conversation dataset (ShareGPT format)
- **Epochs:** 1
- **Learning rate:** 5e-5
- **Batch size:** 2 per device, gradient accumulation 8 (effective batch = 16)
- **Warmup ratio:** 0.05
- **Max sequence length:** 1024
- **Optimizer:** AdamW 8-bit

## Usage

### vLLM (recommended)

```bash
pip install vllm
vllm serve pbanavara/gemma4-transplant-merged --dtype bfloat16
```

### Transformers

```python
import torch
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

model_id = "pbanavara/gemma4-transplant-merged"

processor = AutoProcessor.from_pretrained(model_id)
model = Gemma4ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{"role": "user", "content": [{"type": "text", "text": "Your prompt here"}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)

print(processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Intended Use

This model is intended for transplant coordinator role-play and training simulations. It is **not** intended for clinical decision-making or real patient care.

## Limitations

- Trained on a small private dataset — may overfit to specific conversation patterns
- Not validated for clinical accuracy
- English only

## License

This model is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

import torch
from unsloth import FastLanguageModel

# Load the merged model directly (no vLLM) to verify weights are correct
model, tokenizer = FastLanguageModel.from_pretrained(
    "./gemma4-vllm",
    load_in_4bit=False,
    use_gradient_checkpointing=False,
)

FastLanguageModel.for_inference(model)

messages = [{"role": "user", "content": [{"type": "text", "text": "A patient calls in asking to reschedule their appointment for next Tuesday. How do you handle this?"}]}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

with torch.no_grad():
    outputs = model.generate(input_ids=inputs, max_new_tokens=200, temperature=0.7, do_sample=True)

response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print("Response:", response)

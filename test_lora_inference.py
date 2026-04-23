"""
Test the LoRA adapter directly via Unsloth (no merge) to confirm whether
the training produced good weights before we try to solve the merge issue.
"""
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

ADAPTER_PATH = "./gemma4-transplant/checkpoint-480"

print("Loading base model + LoRA adapter via Unsloth...")
model, processor = FastLanguageModel.from_pretrained(
    ADAPTER_PATH,           # loads base + adapter from checkpoint
    load_in_4bit=False,
    dtype=torch.bfloat16,
)
FastLanguageModel.for_inference(model)

tokenizer = get_chat_template(processor, chat_template="gemma-4")

print("Running inference tests...\n")

def test(label, messages):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ).removeprefix("<bos>")
    inputs = tokenizer(text=text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=120, do_sample=True,
            temperature=0.7, repetition_penalty=1.1,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"=== {label} ===")
    print(response)
    print()

# Test 1: coordinator-initiated
test("COORDINATOR-INITIATED", [
    {"role": "user", "content": (
        "You are an experienced transplant coordinator conducting a post-transplant "
        "follow-up call. You are empathetic, professional, and knowledgeable about "
        "immunosuppression, rejection signs, and post-transplant care protocols.\n\n"
        "[Patient is ready to receive the coordinator's call.]"
    )},
])

# Test 2: patient-initiated
test("PATIENT-INITIATED", [
    {"role": "user", "content": (
        "You are an experienced transplant coordinator. Your goal is to assess the "
        "patient's current status, address concerns, and escalate appropriately."
    )},
    {"role": "assistant", "content": "Transplant coordinator speaking, how can I help you?"},
    {"role": "user", "content": (
        "Hi, this is Sarah. I just saw my creatinine came back at 3.2 mg/dL in the "
        "patient portal and it flagged as high. I'm really scared, should I be worried?"
    )},
])

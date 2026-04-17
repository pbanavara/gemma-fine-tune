from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
import torch
from unsloth.chat_templates import standardize_data_formats
from trl import SFTTrainer
from transformers import TrainingArguments

gemma4_models = [
    # Gemma-4 instruct models:
    "unsloth/gemma-4-E2B-it",
    "unsloth/gemma-4-E4B-it",
    "unsloth/gemma-4-31B-it",
    "unsloth/gemma-4-26B-A4B-it",
    # Gemma-4 base models:
    "unsloth/gemma-4-E2B",
    "unsloth/gemma-4-E4B",
    "unsloth/gemma-4-31B",
    "unsloth/gemma-4-26B-A4B",
] # More models at https://huggingface.co/unsloth

model, processor = FastLanguageModel.from_pretrained(
        "unsloth/gemma-4-E4B-it",
        load_in_4bit=False,
        dtype=torch.bfloat16,
        use_gradient_checkpointing="unsloth"
        )

model = FastLanguageModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 32,                           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 32,                  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,               # We support rank stabilized LoRA
    loftq_config = None,               # And LoftQ
    target_modules = "all-linear",    # Optional now! Can specify a list if needed
)

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    processor,
    chat_template = "gemma-4",
)

from datasets import load_dataset

dataset = load_dataset(
            "json",
            data_files="/home/pbanavara/patient-coordinator-data/conversations_sharegpt.jsonl",
            split="train"
            )

dataset = standardize_data_formats(dataset)   # normalises edge cases
print(f"Dataset {dataset}")

def fix_conversation(convo):
    """Ensure conversation alternates user/assistant, merging consecutive same-role turns."""
    fixed = []
    for msg in convo:
        role = msg.get("role", "")
        # Map system messages to user role (Gemma-4 template doesn't support system role inline)
        if role == "system":
            role = "user"
        content = msg.get("content", "")
        if fixed and fixed[-1]["role"] == role:
            # Merge consecutive same-role messages
            fixed[-1]["content"] += "\n" + content
        else:
            fixed.append({"role": role, "content": content})
    # Drop leading assistant turns — template requires starting with user
    while fixed and fixed[0]["role"] != "user":
        fixed.pop(0)
    return fixed

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = []
    for convo in convos:
        convo = fix_conversation(convo)
        if not convo:
            continue
        text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix('<bos>')
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched = True)


# ── 5. Train ─────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
  model=model,
  tokenizer=tokenizer,
  train_dataset=dataset,
  dataset_text_field="text",
  max_seq_length=2048,
  args=TrainingArguments(
      per_device_train_batch_size=8,       # L40S 48GB handles this comfortably in bf16
      gradient_accumulation_steps=2,       # effective batch = 16, same as before but faster
      num_train_epochs=3,
      learning_rate=2e-4,
      output_dir="./gemma4-transplant",
      fp16=False,
      bf16=True,                           # L40S Ada Lovelace has native bf16 support
      dataloader_num_workers=4,
      optim="adamw_8bit",                  # cuts optimizer memory ~4x, frees headroom for larger batches
  ),
)
trainer.train()

# Merge LoRA adapters into the base weights and save in bfloat16.
# vLLM requires a single merged model — it cannot load a separate LoRA adapter
# on top of a quantized base at inference time.
model.save_pretrained_merged(
    "./gemma4-transplant-vllm",
    processor,
    save_method="merged_16bit",
)

# Gemma Fine-Tune

Fine-tuning Gemma-4 models using [Unsloth](https://github.com/unslothai/unsloth) with LoRA, producing vLLM-compatible checkpoints.

## Model

- **Base model:** `unsloth/gemma-4-E4B-it`
- **Method:** LoRA (rank 32, all-linear target modules)
- **Precision:** bfloat16 (no quantization — ensures vLLM compatibility)

## Hardware

Tested on NVIDIA L40S (48GB VRAM).

## Usage

```bash
pip install unsloth trl transformers datasets
python unsloth-fine-tune.py
```

The script will:
1. Load the base model in bf16
2. Apply LoRA adapters and train on the dataset
3. Merge adapters into base weights and save to `./gemma4-transplant-vllm`

## Inference with vLLM

```bash
vllm serve ./gemma4-transplant-vllm
```

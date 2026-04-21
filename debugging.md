# Gemma-4 Transplant Fine-Tune — Issues & Debugging Log

---

## 1. Gemma-4 Template: System Role Not Supported

**Problem:** Training dataset had `system` → `assistant` → `user` turn ordering (coordinator initiates calls), but Gemma-4's chat template enforces strict `user` / `assistant` alternation and doesn't support inline `system` roles.

**Fix:** `fix_conversation()` — remaps `system` → `user`, merges consecutive same-role turns, drops any leading `assistant` turns so every conversation starts with `user`.

---

## 2. Overfitting — Loss Collapsed to 0.07

**Problem:** 3 epochs at `lr=2e-4` drove training loss to ~0.07, a clear sign of overfitting on a small dataset. Model was memorising conversations rather than generalising.

**Fix:**
- `num_train_epochs` 3 → 1
- `learning_rate` 2e-4 → 5e-5 (also caused catastrophic forgetting on a MoE base — see below)
- Added `warmup_ratio=0.05` for gentle ramp-up

---

## 3. Catastrophic Forgetting on MoE Base

**Problem:** `lr=2e-4` was too aggressive for Gemma-4's Mixture-of-Experts architecture, overwriting base knowledge rather than adapting it.

**Fix:**
- `lr` reduced to `5e-5`
- `r` / `lora_alpha` reduced 32 → 16 (less aggressive rank)
- `lora_dropout` 0 → 0.05 to regularise on small dataset

---

## 4. Training Logs Silent (No Progress Visible)

**Problem:** Default `logging_steps=500` but the run only had ~240 total steps — nothing was ever logged.

**Fix:** `logging_steps=10`

---

## 5. `use_cache=False` Breaking vLLM

**Problem:** Unsloth sets `use_cache=False` in `config.json` during training (required for gradient checkpointing). This flag was baked into the saved model, causing vLLM to fail or behave incorrectly at inference.

**Fix:** Post-save patch in `unsloth-fine-tune.py` and `peft_merge.py` that reads `config.json` and sets `use_cache=True`.

---

## 6. Unsloth's `save_pretrained_merged` Producing Garbage Output

**Problem:** Unsloth's built-in merge produced incoherent output for `Gemma4ForConditionalGeneration` — a known upstream bug with this model architecture.

**Diagnosis:** Wrote `test_lora_inference.py` to test the LoRA adapter *before* merging via Unsloth directly. It produced coherent responses → confirmed weights were good, merge was the problem.

**First fix attempt (`peft_merge.py` v1):** Used standard PEFT `PeftModel.from_pretrained` + `merge_and_unload()` — failed because PEFT doesn't support Unsloth's custom `Gemma4ClippableLinear` layer type:
```
ValueError: Target module Gemma4ClippableLinear(...) is not supported.
```

**Second fix attempt (`peft_merge.py` v2):** Loaded via Unsloth (which handles `Gemma4ClippableLinear`), manually folded `lora_B @ lora_A * scale` deltas into base weights in-place, saved via HF `save_pretrained`. Failed — the PEFT wrapper structure was still serialised, saving `base_layer.weight` keys instead of plain `weight` keys:
```
KeyError: 'layers.0.mlp.down_proj.base_layer.weight'
```

**Final fix (`peft_merge.py` v3):**
- Load **clean** `Gemma4ForConditionalGeneration` with vanilla transformers (no PEFT, no Unsloth)
- Read adapter weights directly from `adapter_model.safetensors`
- Strip the `base_model.model.` prefix from adapter keys to map them to base param names
- Apply `lora_B @ lora_A * scale` (scale = `lora_alpha / r` = 1.0 for checkpoint-480)
- Save with HF `save_pretrained` → clean `weight` keys, no wrapper artifacts

---

## 7. `test_lora_inference.py` — TypeError: NoneType Not Subscriptable

**Problem:** `tokenizer(text, ...)` passed `text` as a positional arg. Unsloth's patched `__call__` doesn't forward positional args correctly, leaving `text=None` inside the Gemma4 processor:
```
TypeError: 'NoneType' object is not subscriptable
  File "processing_gemma4.py", line 130, in __call__
    elif not isinstance(text, list) and not isinstance(text[0], str):
```

**Fix:** Changed to `tokenizer(text=text, ...)` (keyword arg), matching how `peft_merge.py` already called it.

---

## 8. vLLM Model Name Mismatch

**Problem:** vLLM serves the model using its full path as the model ID. Test curls using a short name like `gemma4-transplant` returned 404.

**Fix:** Use the full path as the model name in API calls:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/home/pbanavara/gemma-fine-tune/gemma4-transplant-vllm", ...}'
```

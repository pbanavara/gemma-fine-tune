"""
Push merged model weights to Hugging Face Hub.
Run: python push_to_hub.py
"""
import os
from huggingface_hub import HfApi, create_repo

MODEL_PATH  = "./gemma4-transplant-vllm"
CARD_PATH   = "./MODEL_CARD.md"
REPO_ID     = "pbanavara/gemma4-transplant-merged"  # change as needed
PRIVATE     = True

token = os.environ.get("HF_TOKEN")
if not token:
    raise EnvironmentError("Set HF_TOKEN environment variable before running.")

api = HfApi(token=token)

print(f"Creating repo {REPO_ID} (private={PRIVATE})...")
create_repo(repo_id=REPO_ID, repo_type="model", private=PRIVATE, exist_ok=True, token=token)

print(f"Uploading {MODEL_PATH} → {REPO_ID} ...")
api.upload_folder(
    folder_path=MODEL_PATH,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Add merged Gemma-4 LoRA model (checkpoint-480)",
)

print("Uploading model card...")
api.upload_file(
    path_or_fileobj=CARD_PATH,
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Add model card",
)

print(f"\nDone. Model available at: https://huggingface.co/{REPO_ID}")

from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent / "models"

MODELS = {
    "opus":        "Helsinki-NLP/opus-mt-en-hi",
    "indictrans2": "ai4bharat/indictrans2-en-indic-dist-200M",
    "nllb":        "facebook/nllb-200-distilled-600M",
    "mbart50":     "facebook/mbart-large-50-one-to-many-mmt",
    "qwen":        "Qwen/Qwen2.5-1.5B-Instruct",
}

IGNORE = ["*.h5", "*.msgpack", "*.ot", "flax_model*", "tf_model*", "rust_model*", "*.onnx"]

for name, repo in MODELS.items():
    print(f"\n=== Downloading {name}: {repo} ===")
    snapshot_download(repo_id=repo, local_dir=ROOT / name, ignore_patterns=IGNORE)

print("\nAll done. Models are in:", ROOT)

import sys, time, gc
from pathlib import Path
import torch
import mt_models

SENTENCES = [
    "I went to the Ganpati visarjan.",
    "I wanted to learn about transformers and Wasserstein distance for my career and job.",
    "The meeting has been moved to Friday at four in the afternoon.",
]

names = sys.argv[1:] or list(mt_models.LOADERS)
Path("outputs").mkdir(exist_ok=True)
lines = []

for name in names:
    t0 = time.time()
    try:
        run = mt_models.LOADERS[name]()
        loaded = time.time() - t0
        results = run(SENTENCES)
    except Exception as e:
        print(f"[{name}] FAILED: {e}")
        lines.append(f"=== {name} FAILED: {e}\n")
        continue
    header = f"=== {name} (load {loaded:.1f}s, total {time.time()-t0:.1f}s) ==="
    print(header)
    lines.append(header)
    for s, o in zip(SENTENCES, results):
        lines.append(f"EN: {s}\nHI: {o}\n")
    del run
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

Path("outputs/test_results.txt").write_text("\n".join(lines), encoding="utf-8")
print("\nOpen outputs\\test_results.txt in VS Code to see the Hindi output.")

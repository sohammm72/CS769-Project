"""
test_models.py - run the 5 local MT models on sentences.

Two ways to use it (run both in the (venv) environment, NOT venv_qe):

1. The quick check on 3 built-in sentences (as before):
     python test_models.py                 all five models
     python test_models.py opus nllb       only these models
   Output: outputs\\test_results.txt (read it in VS Code).

2. A bigger test on sentences from a file, for example the first 100 sentences
   of your test set:
     python test_models.py --input data_test\\test_sentences.txt --n 100
     python test_models.py opus nllb --input data_test\\test_sentences.txt --n 100
   Output: outputs\\local_candidates.csv with columns id, source, then one column per model.
   The id is the line number in the input file, so it matches the id in the gold CSV.
   The file is rewritten after every model, so a crash does not lose finished models.
   If the GPU runs out of memory, add --cpu (slower) or close other programs that use the GPU.
   Rerunning only some models (e.g. `python test_models.py indictrans2 qwen --input ...`) keeps the
   other models' columns, if the sentences are the same.
   The CSV is the input of qe_score.py, qe_fusion.py, combine_baseline.py and evaluate.py.
"""

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import mt_models

try:
    import torch
except ImportError:  # only used to free GPU memory
    torch = None

SENTENCES = [
    "I went to the Ganpati visarjan.",
    "I wanted to learn about transformers and Wasserstein distance for my career and job.",
    "The meeting has been moved to Friday at four in the afternoon.",
]


def read_sentences(path, n):
    """First n non-empty lines of the file, with their line numbers (ids start at 1)."""
    if not Path(path).exists():
        sys.exit("Input file not found: %s" % path)
    items = []
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if s:
                items.append((line_no, s))
                if n and len(items) >= n:
                    break
    if not items:
        sys.exit("No sentences in %s" % path)
    return items


def translate_in_batches(run, sents, batch_size):
    out = []
    for i in range(0, len(sents), batch_size):
        out.extend(run(sents[i:i + batch_size]))
        print("    %d/%d sentences" % (min(i + batch_size, len(sents)), len(sents)))
    return out


def write_csv(path, items, results):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    models = list(results)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "source"] + models)
        for k, (line_no, s) in enumerate(items):
            w.writerow([line_no, s] + [results[m][k] for m in models])


def read_existing(path, items, keep_out):
    """Columns already in the output CSV for the same sentences (so a rerun of one model keeps the others)."""
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or [r["id"] for r in rows] != [str(i) for i, _ in items] or [r["source"] for r in rows] != [s for _, s in items]:
        print("Note: %s holds different sentences, so it will be overwritten." % path)
        return {}
    cols = [c for c in rows[0] if c not in ("id", "source") and c not in keep_out]
    return {c: [r[c] for r in rows] for c in cols}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="*", help="model keys (default: all): %s" % " ".join(mt_models.LOADERS))
    p.add_argument("--input", help="text file with one English sentence per line (default: the 3 built-in sentences)")
    p.add_argument("--n", type=int, default=100, help="with --input: how many sentences from the top of the file (default 100)")
    p.add_argument("--batch-size", type=int, default=16, help="with --input: sentences per batch (lower it if CUDA runs out of memory)")
    p.add_argument("--output", default="outputs/local_candidates.csv", help="with --input: the CSV to write")
    p.add_argument("--cpu", action="store_true", help="run on the CPU instead of the GPU (slower, but works when the GPU is out of memory)")
    args = p.parse_args()

    if args.cpu:
        mt_models.DEVICE = "cpu"  # the loaders read these module settings when they are called
        if torch is not None:
            mt_models.DTYPE = torch.float32
        print("Running on the CPU.")
    names = args.models or list(mt_models.LOADERS)
    bad = [m for m in names if m not in mt_models.LOADERS]
    if bad:
        sys.exit("Unknown model(s): %s. Choose from: %s" % (", ".join(bad), ", ".join(mt_models.LOADERS)))
    Path("outputs").mkdir(exist_ok=True)

    if args.input:
        items = read_sentences(args.input, args.n)
    else:
        items = [(i, s) for i, s in enumerate(SENTENCES, start=1)]
    sents = [s for _, s in items]
    print("%d sentences, models: %s" % (len(sents), ", ".join(names)))

    lines = []
    results = read_existing(args.output, items, set(names)) if args.input else {}
    if results:
        print("Keeping the models already in %s: %s" % (args.output, ", ".join(results)))
    for name in names:
        t0 = time.time()
        run = None
        try:
            run = mt_models.LOADERS[name]()
            loaded = time.time() - t0
            if args.input:
                out = translate_in_batches(run, sents, args.batch_size)
            else:
                out = run(sents)
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
            lines.append(f"=== {name} FAILED: {e}\n")
            run = None
        if run is None:  # free whatever the failed model still holds, so the next one has room
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        header = f"=== {name} (load {loaded:.1f}s, total {time.time()-t0:.1f}s) ==="
        print(header)
        lines.append(header)
        for s, o in zip(sents, out):
            lines.append(f"EN: {s}\nHI: {o}\n")
        results[name] = out
        if args.input:
            write_csv(args.output, items, results)
        del run
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.input:
        if results:
            print("\nSaved: %s  (%d sentences x %d models)" % (args.output, len(sents), len(results)))
        else:
            print("\nNo model finished, so no CSV was written.")
    else:
        Path("outputs/test_results.txt").write_text("\n".join(lines), encoding="utf-8")
        print("\nOpen outputs\\test_results.txt in VS Code to see the Hindi output.")


if __name__ == "__main__":
    main()
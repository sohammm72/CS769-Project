"""
combine_baseline.py - the consensus baseline: for every sentence, pick the one
model translation that the other models agree with most.

Why not plain majority voting? With many models, two Hindi translations are
rarely word-for-word identical, so a plain vote often has no winner. This
script therefore works in two steps for each sentence:

  1. Exact majority: if one translation is produced by at least two models and
     no other translation has as many votes, it wins ("majority").
  2. Otherwise, the translation that is most similar to all the others wins
     ("consensus"). Similarity is chrF++ between two translations, so this is
     minimum-Bayes-risk selection with chrF++ as the utility.

No reference translations and no quality-estimation model are used: this is the
point of comparison for the quality-estimation methods that come later.

Usage:
    pip install sacrebleu
    python combine_baseline.py data\\candidate_translations.csv
    python combine_baseline.py data\\candidate_translations.csv --processes 4   (faster on a big run)
    python combine_baseline.py data\\candidate_translations.csv --models "ModelA" "ModelB"

Input:  the CSV from run_api_pipeline.py (columns: id, source, one column per
        model). Empty cells (a model that has not translated a sentence yet) are
        ignored for that sentence.
Output: data\\baseline_consensus.csv with columns
        id, source, translation, chosen_model, method, agreement
        and a printed table of how often each model's translation was picked.
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from sacrebleu.metrics import CHRF

_CHRF = None


def _similarity(hyp, ref):
    """chrF++ score (0-100) of one translation against another."""
    global _CHRF
    if _CHRF is None:
        _CHRF = CHRF(word_order=2)  # word_order=2 makes it chrF++
    return _CHRF.sentence_score(hyp, [ref]).score


def choose_consensus(candidates):
    """Pick the consensus translation for one sentence.

    candidates: list of (model, text) with non-empty text, in model order.
    Returns (position in that list, method, agreement score 0-100).
    """
    n = len(candidates)
    if n == 1:
        return 0, "single", 100.0

    texts = [t for _, t in candidates]
    counts = Counter(texts)

    # Step 1: exact majority with a single clear winner.
    ranked = counts.most_common()
    top_text, top_count = ranked[0]
    if top_count >= 2 and (len(ranked) == 1 or ranked[1][1] < top_count):
        return texts.index(top_text), "majority", 100.0 * (top_count - 1) / (n - 1)

    # Step 2: the translation most similar to all the others. Identical copies
    # count as a perfect match (100) without being scored again.
    best_pos, best_score = 0, -1.0
    seen = set()
    for pos, text in enumerate(texts):
        if text in seen:
            continue
        seen.add(text)
        total = 100.0 * (counts[text] - 1)
        for other, c in counts.items():
            if other != text:
                total += c * _similarity(text, other)
        score = total / (n - 1)
        if score > best_score:  # strict '>' keeps the earliest model on ties
            best_pos, best_score = pos, score
    return best_pos, "consensus", best_score


def _solve_chunk(chunk):
    """Worker for --processes: solve a list of candidate lists."""
    return [choose_consensus(c) if c else None for c in chunk]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("candidates", nargs="?", default="data/candidate_translations.csv", help="CSV from run_api_pipeline.py")
    p.add_argument("--output", default="data/baseline_consensus.csv")
    p.add_argument("--models", nargs="+", help="only use these model columns (default: all)")
    p.add_argument("--processes", type=int, default=1, help="parallel processes (default 1)")
    args = p.parse_args()

    if not os.path.exists(args.candidates):
        sys.exit("File not found: %s" % args.candidates)

    with open(args.candidates, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    if "id" not in header or "source" not in header:
        sys.exit("This does not look like the pipeline's output: it needs 'id' and 'source' columns.")

    models = [c for c in header if c not in ("id", "source")]
    if args.models:
        unknown = [m for m in args.models if m not in models]
        if unknown:
            sys.exit("Not in the file: %s. Columns are: %s" % (", ".join(unknown), ", ".join(models)))
        models = args.models

    # Models with no translations at all are ignored, with a warning.
    empty = [m for m in models if not any((r[m] or "").strip() for r in rows)]
    if empty:
        print("Ignoring models with no translations yet: " + ", ".join(empty))
        models = [m for m in models if m not in empty]
    if not models:
        sys.exit("No translations found in the file.")

    all_candidates = []
    for r in rows:
        all_candidates.append([(m, r[m].strip()) for m in models if (r[m] or "").strip()])

    print("%d sentences, %d models: %s" % (len(rows), len(models), ", ".join(models)))
    t0 = time.time()
    if args.processes > 1:
        size = max(1, len(all_candidates) // (args.processes * 8))
        chunks = [all_candidates[i : i + size] for i in range(0, len(all_candidates), size)]
        with ProcessPoolExecutor(max_workers=args.processes) as ex:
            results = [x for part in ex.map(_solve_chunk, chunks) for x in part]
    else:
        results = []
        for i, c in enumerate(all_candidates, start=1):
            results.append(choose_consensus(c) if c else None)
            if i % 1000 == 0:
                print("  %d / %d sentences" % (i, len(all_candidates)))
    print("Done in %.0f s." % (time.time() - t0))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    picks = Counter()
    methods = Counter()
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "translation", "chosen_model", "method", "agreement"])
        for r, cands, res in zip(rows, all_candidates, results):
            if res is None:
                w.writerow([r["id"], r["source"], "", "", "none", ""])
                methods["none"] += 1
                continue
            pos, method, score = res
            model, text = cands[pos]
            picks[model] += 1
            methods[method] += 1
            w.writerow([r["id"], r["source"], text, model, method, "%.1f" % score])

    done = sum(picks.values())
    print("\nHow often each model's translation was picked (%d sentences):" % done)
    for m in models:
        share = 100.0 * picks[m] / done if done else 0.0
        print("  %5.1f%%  %5d  %s" % (share, picks[m], m))
    print("\nHow the pick was made: " + ", ".join("%s: %d" % (k, v) for k, v in methods.items()))
    print("\nA model that is almost never picked may be producing poor or odd output,")
    print("so it is worth reading a few of its translations.")
    print("Saved: %s" % args.output)


if __name__ == "__main__":
    main()
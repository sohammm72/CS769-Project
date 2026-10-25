"""
qe_score.py - quality estimation (QE) for every model's candidate translation,
then "QE-reranking": pick, per sentence, the candidate with the best QE score.

QE = a reference-free quality score: it looks only at the English source and the
Hindi candidate (no gold translation) and predicts how good the translation is.
We use CometKiwi, the QE metric used in "Don't Rank, Combine!".

Input : data/candidate_translations.csv  (id, source, one column per model)
        or a .txt file from the local model tests, in this format:
            === opus (load 1.5s, total 2.6s) ===
            EN: <English sentence>
            HI: <Hindi translation>
            (blank line, next EN:/HI: pair, then the next === model === section)
Output: data/qe_scores.csv   id, source, one column per model with its QE score
        data/qe_rerank.csv   id, source, translation, chosen_model, qe_score
                             (works with:  evaluate.py ... --extra QE-rerank=data\\qe_rerank.csv)
        data/qe_cache.jsonl  every score ever computed (never scored twice)

Scorers (--scorer):
  cometkiwi  the real thing. Needs:  pip install unbabel-comet
             A Hugging Face account that has accepted the model's license, then
             `huggingface-cli login`. Default model: Unbabel/wmt22-cometkiwi-da
             (small, can run on a laptop CPU for a few hundred sentences).
             Bilal can run the big one on the cluster:  --model Unbabel/wmt23-cometkiwi-da-xxl
             (needs a GPU, so do not run it on the server yourself).
  demo       NOT a quality metric. A stand-in that scores each candidate by how
             similar it is to the other candidates (it needs no model). It only
             exists so the whole pipeline can be tested and shown without a GPU.
             Never report its numbers as QE results.

Optional check against the gold references (the "validate the QE model" step):
  --gold data/gold_all.csv   prints the correlation between QE scores and the
                             chrF++ of each candidate against the gold, and how
                             often QE's top pick is also the best candidate.

Usage:
    python qe_score.py data\\candidate_translations.csv --scorer demo --limit 50
    python qe_score.py data\\candidate_translations.csv --scorer cometkiwi --limit 20
    python qe_score.py data\\candidate_translations.csv --scorer cometkiwi --gold data\\gold_all.csv
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter

from sacrebleu.metrics import CHRF


# ---------------------------------------------------------------------- #
# Small helpers
# ---------------------------------------------------------------------- #
def read_csv(path):
    if not os.path.exists(path):
        sys.exit("File not found: %s" % path)
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def parse_test_results_txt(path):
    """Read the text format written by the local model tests:

        === opus (load 1.5s, total 2.6s) ===
        EN: <English sentence>
        HI: <Hindi translation>

        EN: ...
        HI: ...
        === nllb (...) ===
        ...

    Returns (header, rows) in the same shape as the candidates CSV:
    columns id, source, then one column per model. Sentences are matched
    across models by their English text.
    """
    if not os.path.exists(path):
        sys.exit("File not found: %s" % path)
    models, sources = [], []
    table = {}  # (model, english) -> hindi
    model, pending_en = None, None
    with open(path, encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("===") and line.endswith("==="):
                inner = line.strip("= ").strip()
                model = inner.split(" (")[0].strip()
                if model not in models:
                    models.append(model)
                pending_en = None
            elif line.startswith("EN:") and model is not None:
                pending_en = line[3:].strip()
                if pending_en not in sources:
                    sources.append(pending_en)
            elif line.startswith("HI:") and model is not None and pending_en is not None:
                table[(model, pending_en)] = line[3:].strip()
                pending_en = None
    if not models or not sources:
        sys.exit("No '=== model ===' sections with EN:/HI: lines found in %s." % path)
    rows = []
    for i, en in enumerate(sources, start=1):
        row = {"id": str(i), "source": en}
        for m in models:
            row[m] = table.get((m, en), "")
        rows.append(row)
    return ["id", "source"] + models, rows


def load_candidates(path):
    """CSV from run_api_pipeline.py, or the text format of the local model tests."""
    if path.lower().endswith(".txt"):
        return parse_test_results_txt(path)
    return read_csv(path)


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def ranks(values):
    """Ranks with ties averaged (for Spearman)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


# ---------------------------------------------------------------------- #
# Scorers
# ---------------------------------------------------------------------- #
def load_cache(path):
    cache = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    cache[(r["scorer"], r["src"], r["mt"])] = r["score"]
                except (ValueError, KeyError):
                    continue  # half-written last line
    return cache


def score_with_cometkiwi(pairs, model_name, batch_size, gpus):
    """pairs: list of (src, mt). Returns a list of scores in the same order."""
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as e:
        sys.exit(
            "Could not import the 'comet' package (from unbabel-comet). The real error was:\n"
            "    %s: %s\n"
            "If the package is missing, install it with:  pip install unbabel-comet\n"
            "(it installs PyTorch too, which is large; use Python 3.11 or 3.12, not 3.13).\n"
            "Use --scorer demo to test the pipeline without it." % (type(e).__name__, e)
        )
    print("Loading %s (the first time this downloads the model)..." % model_name)
    # Download with huggingface_hub first: COMET hides the real reason of a failed
    # download behind "not supported by COMET", and the real reason is what we need.
    if not os.path.exists(model_name):
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=model_name)
        except Exception as e:
            sys.exit(
                "Could not download %s from Hugging Face.\nThe real error was:\n    %s: %s\n"
                "If it mentions 'gated', '401' or '403': accept the model's license on its page "
                "with the same account, and make sure your token may read gated repos "
                "(fine-grained tokens need that permission ticked)." % (model_name, type(e).__name__, e)
            )
    try:
        checkpoint = model_name if model_name.endswith(".ckpt") else download_model(model_name)
        model = load_from_checkpoint(checkpoint)
    except Exception as e:
        sys.exit("Could not load %s: %s: %s" % (model_name, type(e).__name__, e))
    data = [{"src": s, "mt": t} for s, t in pairs]
    try:
        # num_workers=0: COMET warns that worker processes can break on Windows
        # (and a first run crashed with an access violation while using 1).
        out = model.predict(data, batch_size=batch_size, gpus=gpus, num_workers=0)
    except TypeError:  # older versions have no num_workers
        out = model.predict(data, batch_size=batch_size, gpus=gpus)
    scores = out.scores if hasattr(out, "scores") else out[0]
    return [float(s) for s in scores]


def demo_scores(cands, chrf):
    """Stand-in 'score': average chrF++ similarity to the other candidates (0-100 -> 0-1)."""
    out = []
    for i, t in enumerate(cands):
        others = [o for j, o in enumerate(cands) if j != i]
        out.append(sum(chrf.sentence_score(t, [o]).score for o in others) / len(others) / 100.0 if others else 1.0)
    return out


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("candidates", nargs="?", default="data/candidate_translations.csv")
    p.add_argument("--scorer", choices=["cometkiwi", "demo"], default="cometkiwi")
    p.add_argument("--model", default="Unbabel/wmt22-cometkiwi-da", help="CometKiwi checkpoint name")
    p.add_argument("--models", nargs="+", help="only these model columns (default: all)")
    p.add_argument("--limit", type=int, help="only the first N sentences")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gpus", type=int, default=0, help="0 = CPU (default)")
    p.add_argument("--gold", help="gold CSV (id, source, gold_reference) to check QE against")
    p.add_argument("--cache", default="data/qe_cache.jsonl")
    p.add_argument("--out-scores", default="data/qe_scores.csv")
    p.add_argument("--out-pick", default="data/qe_rerank.csv")
    args = p.parse_args()

    header, rows = load_candidates(args.candidates)
    if "id" not in header or "source" not in header:
        sys.exit("The candidates file needs 'id' and 'source' columns.")
    if args.limit:
        rows = rows[: args.limit]
    models = [c for c in header if c not in ("id", "source")]
    if args.models:
        bad = [m for m in args.models if m not in models]
        if bad:
            sys.exit("Not in the file: %s" % ", ".join(bad))
        models = args.models
    models = [m for m in models if any((r[m] or "").strip() for r in rows)]
    if not models:
        sys.exit("No translations in the file yet.")
    scorer_key = args.scorer if args.scorer == "demo" else args.model
    print("%d sentences, %d models, scorer: %s" % (len(rows), len(models), scorer_key))

    # ---- scores: score[(id, model)] ------------------------------------ #
    score = {}
    if args.scorer == "demo":
        chrf = CHRF(word_order=2)
        for r in rows:
            present = [m for m in models if (r[m] or "").strip()]
            vals = demo_scores([r[m].strip() for m in present], chrf)
            for m, v in zip(present, vals):
                score[(r["id"], m)] = v
        print("NOTE: 'demo' is not a quality metric. It only tests the pipeline.")
    else:
        cache = load_cache(args.cache)
        todo, seen = [], set()
        for r in rows:
            for m in models:
                mt = (r[m] or "").strip()
                key = (scorer_key, r["source"].strip(), mt)
                if mt and key not in cache and key not in seen:
                    seen.add(key)
                    todo.append((key[1], key[2]))
        print("%d (source, translation) pairs to score; %d already in the cache." % (len(todo), len(cache)))
        if todo:
            new = score_with_cometkiwi(todo, args.model, args.batch_size, args.gpus)
            os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)
            with open(args.cache, "a", encoding="utf-8") as f:
                for (s, t), v in zip(todo, new):
                    cache[(scorer_key, s, t)] = v
                    f.write(json.dumps({"scorer": scorer_key, "src": s, "mt": t, "score": v}, ensure_ascii=False) + "\n")
        for r in rows:
            for m in models:
                mt = (r[m] or "").strip()
                if mt:
                    score[(r["id"], m)] = cache[(scorer_key, r["source"].strip(), mt)]

    # ---- outputs -------------------------------------------------------- #
    os.makedirs(os.path.dirname(os.path.abspath(args.out_scores)), exist_ok=True)
    picks = Counter()
    sums, counts = Counter(), Counter()
    with open(args.out_scores, "w", encoding="utf-8-sig", newline="") as fs, \
         open(args.out_pick, "w", encoding="utf-8-sig", newline="") as fp:
        ws, wp = csv.writer(fs), csv.writer(fp)
        ws.writerow(["id", "source"] + models)
        wp.writerow(["id", "source", "translation", "chosen_model", "qe_score"])
        for r in rows:
            ws.writerow([r["id"], r["source"]] + [("%.4f" % score[(r["id"], m)]) if (r["id"], m) in score else "" for m in models])
            avail = [(score[(r["id"], m)], m) for m in models if (r["id"], m) in score]
            for v, m in avail:
                sums[m] += v
                counts[m] += 1
            if avail:
                best_v, best_m = max(avail, key=lambda x: x[0])  # ties: first max wins
                picks[best_m] += 1
                wp.writerow([r["id"], r["source"], r[best_m].strip(), best_m, "%.4f" % best_v])
            else:
                wp.writerow([r["id"], r["source"], "", "", ""])

    n_pick = sum(picks.values())
    print("\nPer model: average QE score, and how often its translation had the top QE score")
    for m in models:
        avg = sums[m] / counts[m] if counts[m] else float("nan")
        print("  avg %.3f   top-scored %5.1f%%   %s" % (avg, 100.0 * picks[m] / max(n_pick, 1), m))
    print("\nSaved: %s, %s" % (args.out_scores, args.out_pick))

    # ---- optional: check QE against the gold references ----------------- #
    if args.gold:
        g_header, g_rows = read_csv(args.gold)
        gold = {g["id"]: (g.get("gold_reference") or "").strip() for g in g_rows}
        gold_src = {g["id"]: (g.get("source") or "").strip() for g in g_rows}
        chrf = CHRF(word_order=2)
        xs, ys = [], []
        top1_hits, top1_n, pick_chrf, mean_chrf, best_chrf = 0, 0, [], [], []
        for r in rows:
            ref = gold.get(r["id"], "")
            if not ref:
                continue
            if gold_src.get(r["id"]) and gold_src[r["id"]] != r["source"].strip():
                sys.exit("Gold source differs from the candidates for id %s: wrong gold file?" % r["id"])
            per = []
            for m in models:
                if (r["id"], m) in score:
                    c = chrf.sentence_score(r[m].strip(), [ref]).score
                    per.append((score[(r["id"], m)], c, m))
                    xs.append(score[(r["id"], m)])
                    ys.append(c)
            if len(per) >= 2:
                qe_top = max(per, key=lambda t: t[0])
                best = max(per, key=lambda t: t[1])
                top1_n += 1
                top1_hits += (qe_top[2] == best[2]) or (qe_top[1] >= best[1])
                pick_chrf.append(qe_top[1])
                mean_chrf.append(sum(t[1] for t in per) / len(per))
                best_chrf.append(best[1])
        if top1_n:
            print("\nQE checked against the gold references (%d sentences, %d candidates):" % (top1_n, len(xs)))
            print("  Pearson  correlation QE vs chrF++: %.3f" % pearson(xs, ys))
            print("  Spearman correlation QE vs chrF++: %.3f" % spearman(xs, ys))
            print("  QE's top pick is the best candidate (or ties it): %.1f%%" % (100.0 * top1_hits / top1_n))
            print("  average chrF++:  QE pick %.1f | random pick (mean of models) %.1f | oracle %.1f"
                  % (sum(pick_chrf) / top1_n, sum(mean_chrf) / top1_n, sum(best_chrf) / top1_n))
        else:
            print("\nNo sentences with a gold reference among the scored ones.")


if __name__ == "__main__":
    main()
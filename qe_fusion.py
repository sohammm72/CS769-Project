"""
qe_fusion.py - QE-fusion from "Don't Rank, Combine!" (Vernikos & Popescu-Belis, 2024),
Algorithm 1 on page 4 of the paper.

Instead of picking ONE candidate (reranking), build a new sentence from pieces of
several candidates, guided by a quality-estimation (QE) score:

  1. Score every candidate with QE. The top-scored one is the BASE.
  2. Compare the base with each other candidate word by word (Python's difflib).
     Every place where they differ is a "divergent span" (words replaced, added
     or removed). Spans that overlap are merged into one SLOT, whose options are
     the base's own words plus each different version found in the other candidates.
  3. Go through the slots from left to right. For every hypothesis in the BEAM,
     try each option of the slot. This creates new sentences. Score them all with
     QE, keep the best `beam` (the beam), and move to the next slot.
  4. The best sentence in the beam at the end is the fused translation.

Efficiency (paper, appendix A.1): all sentences are processed step by step
together so the QE model is called in big batches, every (source, translation)
pair is scored only once (cache), and sentences with no slots left stop early.

Differences from the paper's pseudo-code (which uses string replace):
  * edits are made by word position, so a phrase that occurs twice is safe;
  * overlapping spans from different candidates are merged into one slot;
  * punctuation (including the Hindi danda) is split off words before comparing,
    and joined again afterwards, so "है।" and "है ।" do not count as a difference.
Our candidates come from DIFFERENT MODELS, not from sampling one model as in the
paper. Use --top-k (default 5, as in the paper's pools) to keep only the best
candidates by QE, so the spans stay small and the cost stays low.

Input : data/candidate_translations.csv (id, source, one column per model),
        or the .txt format of the local model tests (as in qe_score.py)
Output: data/qe_fusion.csv with columns
        id, source, translation, base_model, base_qe, fused_qe, changed, novel,
        slots, hypotheses_scored
        ("novel" = the fused sentence is not identical to any candidate.)
        Works with:  evaluate.py ... --extra QE-fusion=data\\qe_fusion.csv

Scorers (--scorer):
  cometkiwi  the real QE metric (needs: pip install unbabel-comet, and a Hugging
             Face login with the model's license accepted). Same cache file as
             qe_score.py, so scores are never computed twice.
  demo       NOT a quality metric: scores a sentence by its similarity to the
             candidates. Only for testing that the pipeline runs.

Usage:
    python qe_fusion.py --selftest                      (the paper's Figure 1 example, no model needed)
    python qe_fusion.py data\\candidate_translations.csv --scorer demo --limit 50
    python qe_fusion.py data\\candidate_translations.csv --scorer cometkiwi --limit 200
"""

import argparse
import csv
import difflib
import json
import os
import sys
import time

from sacrebleu.metrics import CHRF

from qe_score import load_cache, load_candidates

try:
    sys.stdout.reconfigure(encoding="utf-8")  # so Hindi prints correctly on Windows
except Exception:
    pass

# Punctuation that is split off words as separate tokens (the Hindi danda included).
PEEL = set('.,;:!?।॥()[]{}"“”‘’…')
ATTACH_LEFT = set(".,;:!?।॥)]}”’…")  # no space before these
OPENERS = set("([{“‘")  # no space after these


# ---------------------------------------------------------------------- #
# Tokens
# ---------------------------------------------------------------------- #
def tokenize(text):
    """Words, with leading/trailing punctuation split off as their own tokens."""
    tokens = []
    for w in text.split():
        lead, trail = [], []
        while w and w[0] in PEEL:
            lead.append(w[0])
            w = w[1:]
        while w and w[-1] in PEEL:
            trail.append(w[-1])
            w = w[:-1]
        tokens.extend(lead)
        if w:
            tokens.append(w)
        tokens.extend(reversed(trail))
    return tokens


def detokenize(tokens):
    """Join tokens back into a sentence (no space before closing punctuation)."""
    out = []
    quote_open = False
    for i, t in enumerate(tokens):
        if i == 0:
            out.append(t)
            if t == '"':
                quote_open = True
            continue
        prev = tokens[i - 1]
        if t == '"':
            if quote_open:  # closing quote
                out.append(t)
                quote_open = False
            else:  # opening quote
                out.append(" " + t)
                quote_open = True
            continue
        if t in ATTACH_LEFT or prev in OPENERS or (prev == '"' and quote_open):
            out.append(t)
        else:
            out.append(" " + t)
    return "".join(out)


# ---------------------------------------------------------------------- #
# Slots: where the candidates differ from the base
# ---------------------------------------------------------------------- #
def build_slots(base, others):
    """Find the divergent spans between `base` and every other candidate.

    base, others: lists of tokens. Returns a list of slots (start, end, options)
    sorted by position. options[0] is the base's own tokens for that stretch;
    the rest are alternative versions found in the other candidates.
    """
    edits = []  # (start, end, candidate number, replacement tokens)
    for ci, cand in enumerate(others):
        matcher = difflib.SequenceMatcher(None, base, cand, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                edits.append((i1, i2, ci, tuple(cand[j1:j2])))
    edits.sort(key=lambda e: (e[0], e[1], e[2]))

    # Merge edits that overlap in the base into regions.
    regions = []  # [start, end, [edits]]
    for e in edits:
        i1, i2 = e[0], e[1]
        if regions:
            a, b, lst = regions[-1]
            same_insertion_point = (a == b == i1 == i2)
            if i1 < b or same_insertion_point:
                regions[-1][1] = max(b, i2)
                lst.append(e)
                continue
        regions.append([i1, i2, [e]])

    slots = []
    for a, b, lst in regions:
        base_seg = tuple(base[a:b])
        options = [base_seg]
        by_candidate = {}
        for i1, i2, ci, repl in lst:
            by_candidate.setdefault(ci, []).append((i1, i2, repl))
        for ci in sorted(by_candidate):
            seg, pos = [], a
            for i1, i2, repl in sorted(by_candidate[ci]):
                seg.extend(base[pos:i1])
                seg.extend(repl)
                pos = i2
            seg.extend(base[pos:b])
            seg = tuple(seg)
            if seg not in options:
                options.append(seg)
        if len(options) > 1:
            slots.append((a, b, options))
    return slots


def assemble(base, slots, config):
    """Tokens of the sentence where slot k uses option config[k]."""
    out, pos = [], 0
    for (a, b, options), choice in zip(slots, config):
        out.extend(base[pos:a])
        out.extend(options[choice])
        pos = b
    out.extend(base[pos:])
    return out


# ---------------------------------------------------------------------- #
# Scorers (anything with score_many(list of (source, translation)) -> scores)
# ---------------------------------------------------------------------- #
class FuncScorer:
    """Wraps a plain function fn(source, translation) -> score. Caches results."""

    def __init__(self, fn):
        self.fn = fn
        self.cache = {}
        self.calls = 0

    def score_many(self, pairs):
        out = []
        for src, mt in pairs:
            if (src, mt) not in self.cache:
                self.cache[(src, mt)] = self.fn(src, mt)
                self.calls += 1
            out.append(self.cache[(src, mt)])
        return out


class CometKiwiScorer:
    """The real QE metric. Loads the model once, and shares qe_score.py's cache file."""

    def __init__(self, model_name, batch_size, gpus, cache_path):
        self.key = model_name
        self.batch_size = batch_size
        self.gpus = gpus
        self.cache_path = cache_path
        self.cache = load_cache(cache_path)
        self.model = None
        self.calls = 0

    def _load(self):
        if self.model is not None:
            return
        try:
            from comet import download_model, load_from_checkpoint
        except ImportError as e:
            sys.exit(
                "Could not import the 'comet' package (from unbabel-comet). The real error was:\n"
                "    %s: %s\n"
                "Run this in the QE environment: (venv_qe), Python 3.12 (README Step 11).\n"
                "Use --scorer demo to test the pipeline without it." % (type(e).__name__, e)
            )
        print("Loading %s (the first time this downloads the model)..." % self.key)
        # Download with huggingface_hub first so a gated-model error is shown as it really is.
        if not os.path.exists(self.key):
            try:
                from huggingface_hub import snapshot_download

                snapshot_download(repo_id=self.key)
            except Exception as e:
                sys.exit(
                    "Could not download %s from Hugging Face.\nThe real error was:\n    %s: %s\n"
                    "If it mentions 'gated', '401' or '403': accept the model's license on its page "
                    "with the same account as your token." % (self.key, type(e).__name__, e)
                )
        try:
            checkpoint = self.key if self.key.endswith(".ckpt") else download_model(self.key)
            self.model = load_from_checkpoint(checkpoint)
        except Exception as e:
            sys.exit("Could not load %s: %s: %s" % (self.key, type(e).__name__, e))

    def score_many(self, pairs):
        need, seen = [], set()
        for s, t in pairs:
            k = (self.key, s, t)
            if k not in self.cache and k not in seen:
                seen.add(k)
                need.append((s, t))
        if need:
            self._load()
            data = [{"src": s, "mt": t} for s, t in need]
            try:
                out = self.model.predict(data, batch_size=self.batch_size, gpus=self.gpus, num_workers=0)
            except TypeError:
                out = self.model.predict(data, batch_size=self.batch_size, gpus=self.gpus)
            scores = out.scores if hasattr(out, "scores") else out[0]
            os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                for (s, t), v in zip(need, scores):
                    self.cache[(self.key, s, t)] = float(v)
                    f.write(json.dumps({"scorer": self.key, "src": s, "mt": t, "score": float(v)}, ensure_ascii=False) + "\n")
            self.calls += len(need)
        return [self.cache[(self.key, s, t)] for s, t in pairs]


# ---------------------------------------------------------------------- #
# The algorithm
# ---------------------------------------------------------------------- #
def fuse(items, scorer, beam=5, top_k=5, max_slots=20, log=None):
    """QE-fusion for many sentences at once.

    items: list of dicts {id, source, cands: [(model, text), ...]} (non-empty texts).
    Returns a list of result dicts, one per item, in the same order.
    """
    # 1. QE score of every candidate (one big batch), then pick the base.
    scorer.score_many([(it["source"], t) for it in items for _, t in it["cands"]])
    states = []
    for it in items:
        uniq, seen = [], set()
        for m, t in it["cands"]:
            if t not in seen:
                seen.add(t)
                uniq.append((m, t))
        qes = scorer.score_many([(it["source"], t) for _, t in uniq])
        order = sorted(range(len(uniq)), key=lambda i: (-qes[i], i))  # ties: earlier model first
        pool = [uniq[i] for i in order]
        pool_qe = [qes[i] for i in order]
        if top_k:
            pool, pool_qe = pool[:top_k], pool_qe[:top_k]
        base_model, base_text = pool[0]
        base_tokens = tokenize(base_text)
        slots = build_slots(base_tokens, [tokenize(t) for _, t in pool[1:]])
        n_slots_total = len(slots)
        slots = slots[:max_slots]
        states.append({
            "it": it, "base_model": base_model, "base_text": base_text, "base_qe": pool_qe[0],
            "base_tokens": base_tokens, "slots": slots, "n_slots": n_slots_total,
            "beam": [(tuple([0] * len(slots)), base_text, pool_qe[0])],
            "all_tokens": {tuple(tokenize(t)) for _, t in it["cands"]},
            "scored": 0,
        })

    # 2. Slot by slot, over all sentences at once.
    steps = max((len(s["slots"]) for s in states), default=0)
    for t in range(steps):
        batch = []
        for st in states:
            if t >= len(st["slots"]):
                continue  # early exit: nothing left to try for this sentence
            options = st["slots"][t][2]
            known = {text for _, text, _ in st["beam"]}
            new = []
            for cfg, _, _ in st["beam"]:
                for k in range(1, len(options)):
                    cfg2 = cfg[:t] + (k,) + cfg[t + 1:]
                    text2 = detokenize(assemble(st["base_tokens"], st["slots"], cfg2))
                    if text2 not in known:
                        known.add(text2)
                        new.append((cfg2, text2))
            st["new"] = new
            batch.extend((st["it"]["source"], text2) for _, text2 in new)
        scores = iter(scorer.score_many(batch))
        for st in states:
            new = st.pop("new", None)
            if new is None:
                continue
            scored = [(cfg2, text2, next(scores)) for cfg2, text2 in new]
            st["scored"] += len(scored)
            merged = st["beam"] + scored  # the old hypotheses stay in the running
            merged.sort(key=lambda x: -x[2])  # stable: ties keep the older hypothesis
            st["beam"] = merged[:beam]
        if log:
            log("  slot %d of %d done (%d hypotheses scored this step)" % (t + 1, steps, len(batch)))

    # 3. Collect results.
    results = []
    for st in states:
        cfg, text, score = st["beam"][0]
        changed = any(cfg)
        final_text = text if changed else st["base_text"]
        results.append({
            "id": st["it"]["id"], "source": st["it"]["source"], "translation": final_text,
            "base_model": st["base_model"], "base_qe": st["base_qe"], "fused_qe": score,
            "changed": "yes" if changed else "no",
            "novel": "yes" if tuple(tokenize(final_text)) not in st["all_tokens"] else "no",
            "slots": st["n_slots"], "hypotheses_scored": st["scored"],
        })
    return results


# ---------------------------------------------------------------------- #
# Self-test: the example from Figure 1 of the paper
# ---------------------------------------------------------------------- #
def selftest():
    print("Self-test: Figure 1 of the paper (German source 'Brand in franzoesischem Chemiewerk geloescht').")
    cands = [
        ("hyp1", "Fire in French chemical plant extinguished"),
        ("hyp2", "Fire extinguished on French chemical plant"),
        ("hyp3", "Fire at French chemical plant cleared"),
    ]
    ideal = "Fire at French chemical plant extinguished"
    ideal_t = tokenize(ideal)

    def toy_qe(src, text):  # a toy QE that knows the ideal sentence (for the test only)
        return difflib.SequenceMatcher(None, tokenize(text), ideal_t, autojunk=False).ratio()

    scorer = FuncScorer(toy_qe)
    base = tokenize(cands[0][1])
    slots = build_slots(base, [tokenize(c[1]) for c in cands[1:]])
    print("Base:", cands[0][1])
    for a, b, options in slots:
        print("  slot over words %d-%d: %s" % (a, b, " | ".join(" ".join(o) or "(nothing)" for o in options)))
    res = fuse([{"id": "1", "source": "src", "cands": cands}], scorer, beam=5, top_k=0)[0]
    print("Fused :", res["translation"])
    print("Ideal :", ideal)
    assert res["translation"] == ideal, "self-test failed"
    assert res["novel"] == "yes" and res["changed"] == "yes"
    print("OK - the fused sentence is new: it takes 'at' from hyp3 and 'extinguished' from hyp1.")


# ---------------------------------------------------------------------- #
# Command line
# ---------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("candidates", nargs="?", default="data/candidate_translations.csv")
    p.add_argument("--selftest", action="store_true", help="run the Figure 1 example and exit")
    p.add_argument("--scorer", choices=["cometkiwi", "demo"], default="cometkiwi")
    p.add_argument("--model", default="Unbabel/wmt22-cometkiwi-da", help="CometKiwi checkpoint name")
    p.add_argument("--models", nargs="+", help="only these model columns (default: all)")
    p.add_argument("--limit", type=int, help="only the first N sentences")
    p.add_argument("--beam", type=int, default=5, help="beam size b (default 5)")
    p.add_argument("--top-k", type=int, default=5, help="use only the k best candidates by QE (default 5, 0 = all)")
    p.add_argument("--max-slots", type=int, default=20, help="at most this many divergent spans per sentence (default 20)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gpus", type=int, default=0, help="0 = CPU (default)")
    p.add_argument("--cache", default="data/qe_cache.jsonl")
    p.add_argument("--output", default="data/qe_fusion.csv")
    args = p.parse_args()

    if args.selftest:
        selftest()
        return

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
    items = []
    for r in rows:
        cands = [(m, r[m].strip()) for m in models if (r[m] or "").strip()]
        if cands:
            items.append({"id": r["id"], "source": r["source"].strip(), "cands": cands})
    if not items:
        sys.exit("No translations in the file yet.")

    if args.scorer == "demo":
        chrf = CHRF(word_order=2)
        pools = {it["source"]: [t for _, t in it["cands"]] for it in items}

        def demo_fn(src, text):
            others = [c for c in pools.get(src, []) if c != text]
            return sum(chrf.sentence_score(text, [o]).score for o in others) / len(others) / 100.0 if others else 1.0

        scorer = FuncScorer(demo_fn)
        print("NOTE: 'demo' is not a quality metric. It only tests the pipeline.")
    else:
        scorer = CometKiwiScorer(args.model, args.batch_size, args.gpus, args.cache)

    print("%d sentences, %d models, beam %d, top-k %s" % (len(items), len(models), args.beam, args.top_k or "all"))
    t0 = time.time()
    results = fuse(items, scorer, beam=args.beam, top_k=args.top_k, max_slots=args.max_slots, log=print)
    secs = time.time() - t0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    cols = ["id", "source", "translation", "base_model", "base_qe", "fused_qe", "changed", "novel", "slots", "hypotheses_scored"]
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([("%.4f" % r[c]) if c in ("base_qe", "fused_qe") else r[c] for c in cols])

    n = len(results)
    print("\nDone in %.0f s. Saved: %s" % (secs, args.output))
    print("  average QE of the base candidate : %.4f" % (sum(r["base_qe"] for r in results) / n))
    print("  average QE of the fused sentence : %.4f" % (sum(r["fused_qe"] for r in results) / n))
    print("  sentences changed by fusion      : %.1f%%" % (100.0 * sum(r["changed"] == "yes" for r in results) / n))
    print("  fused sentence is new (not a candidate): %.1f%%" % (100.0 * sum(r["novel"] == "yes" for r in results) / n))
    print("  average divergent spans per sentence  : %.1f" % (sum(r["slots"] for r in results) / n))
    print("  hypotheses scored in total            : %d (%d new scorer calls)" % (sum(r["hypotheses_scored"] for r in results), scorer.calls))


if __name__ == "__main__":
    main()
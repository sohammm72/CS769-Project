"""
evaluate.py - score every model, the consensus baseline, and an "oracle"
against your gold (human) Hindi references, using chrF++ and BLEU.

What the rows of the result table mean
  * each model:   that model's own translations
  * baseline:     the consensus pick from combine_baseline.py, computed here
                  on the same sentences
  * oracle:       for every sentence, the best of the models' translations,
                  chosen by peeking at the gold reference. You cannot do this
                  for real, but it shows the best any method that only picks
                  one model's translation could possibly achieve, so it tells
                  you how much room a combining method has.
  * extras:       any other method's output, added with --extra (for example
                  the quality-estimation pick or the fused translations later)

Only sentences that have a gold reference are scored, and only sentences that
every listed system has a translation for, so all rows of the table are
compared on exactly the same sentences (the count is printed as n).

chrF++ is the more reliable number for Hindi. BLEU is included because it is
the usual one to report, but it is harsh on a morphologically rich language.

Usage:
    pip install sacrebleu
    python evaluate.py data\\candidate_translations.csv data\\gold_template.csv
    python evaluate.py data\\candidate_translations.csv data\\gold_template.csv --extra QE=data\\qe_pick.csv

Gold file: columns id, (source), gold_reference. Rows with an empty
gold_reference are skipped. If the gold file has a 'source' column it is
checked against the candidates, to catch a gold file made from a different
sentence list.
Extra file: columns id and translation (the baseline file qualifies).
Output: a printed table, also saved to data\\evaluation_results.csv.
"""

import argparse
import csv
import os
import sys
from collections import Counter

from sacrebleu.metrics import BLEU, CHRF

from combine_baseline import choose_consensus


def read_csv(path):
    if not os.path.exists(path):
        sys.exit("File not found: %s" % path)
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("candidates", help="CSV from run_api_pipeline.py")
    p.add_argument("gold", help="gold CSV (id, source, gold_reference)")
    p.add_argument("--extra", nargs="*", default=[], metavar="NAME=FILE", help="other methods' outputs to include (CSV with id and translation)")
    p.add_argument("--models", nargs="+", help="only these model columns (default: all)")
    p.add_argument("--output", default="data/evaluation_results.csv")
    p.add_argument("--allow-source-mismatch", action="store_true", help="do not stop if the gold sources differ from the candidates")
    args = p.parse_args()

    # ---- candidates and gold ------------------------------------------- #
    header, cand_rows = read_csv(args.candidates)
    if "id" not in header or "source" not in header:
        sys.exit("The candidates file needs 'id' and 'source' columns.")
    models = [c for c in header if c not in ("id", "source")]
    if args.models:
        unknown = [m for m in args.models if m not in models]
        if unknown:
            sys.exit("Not in the candidates file: %s" % ", ".join(unknown))
        models = args.models
    cand = {r["id"]: r for r in cand_rows}

    g_header, g_rows = read_csv(args.gold)
    if "id" not in g_header or "gold_reference" not in g_header:
        sys.exit("The gold file needs 'id' and 'gold_reference' columns.")
    gold = {}
    for r in g_rows:
        ref = (r["gold_reference"] or "").strip()
        if ref:
            gold[r["id"]] = (ref, (r.get("source") or "").strip())
    if not gold:
        sys.exit("No gold references are filled in yet (the gold_reference column is empty).")

    # Catch a gold file that belongs to a different sentence list.
    mismatches = [
        i for i, (_, src) in gold.items()
        if src and i in cand and cand[i]["source"].strip() != src
    ]
    if mismatches and not args.allow_source_mismatch:
        sys.exit(
            "%d gold rows have a different English sentence than the candidates file "
            "(for example id %s). The gold file was probably made from another sentence "
            "list, for instance before make_sample.py was rerun. Make the gold file again, "
            "or use --allow-source-mismatch if you know it is fine."
            % (len(mismatches), mismatches[0])
        )

    ids = [i for i in gold if i in cand]
    print("Gold references filled in: %d (%d of them found in the candidates file)." % (len(gold), len(ids)))

    # ---- extras --------------------------------------------------------- #
    extras = {}
    for spec in args.extra:
        if "=" not in spec:
            sys.exit("--extra needs NAME=FILE, got '%s'." % spec)
        name, path = spec.split("=", 1)
        e_header, e_rows = read_csv(path)
        if "id" not in e_header or "translation" not in e_header:
            sys.exit("Extra file %s needs 'id' and 'translation' columns." % path)
        extras[name] = {r["id"]: (r["translation"] or "").strip() for r in e_rows}

    # ---- keep models that have data, and the common set of sentences ---- #
    usable = [m for m in models if any((cand[i][m] or "").strip() for i in ids)]
    for m in models:
        if m not in usable:
            print("Ignoring %s: no translations for the gold sentences." % m)
    models = usable
    if not models:
        sys.exit("None of the models has translations for the gold sentences yet.")

    common = [
        i for i in ids
        if all((cand[i][m] or "").strip() for m in models)
        and all(extras[n].get(i) for n in extras)
    ]
    print("Scoring %d sentences that have a gold reference and a translation from every system." % len(common))
    if len(common) < len(ids):
        print("(%d gold sentences were left out because some system has no translation for them yet.)" % (len(ids) - len(common)))
    if not common:
        sys.exit("No sentence has both a gold reference and translations from all systems.")

    refs = [gold[i][0] for i in common]

    # ---- systems -------------------------------------------------------- #
    systems = {}
    for m in models:
        systems[m] = [cand[i][m].strip() for i in common]

    baseline, oracle = [], []
    oracle_picks = Counter()
    chrf = CHRF(word_order=2)
    for i, ref in zip(common, refs):
        cands = [(m, cand[i][m].strip()) for m in models]
        pos, _, _ = choose_consensus(cands)
        baseline.append(cands[pos][1])
        best_m, best_t, best_s = None, None, -1.0
        for m, t in cands:
            s = chrf.sentence_score(t, [ref]).score
            if s > best_s:
                best_m, best_t, best_s = m, t, s
        oracle.append(best_t)
        oracle_picks[best_m] += 1
    systems["baseline (consensus)"] = baseline
    for name, table in extras.items():
        systems[name] = [table[i] for i in common]
    systems["oracle (best model per sentence)"] = oracle

    # ---- scores --------------------------------------------------------- #
    bleu = BLEU()
    rows = []
    for name, hyps in systems.items():
        rows.append(
            (
                name,
                chrf.corpus_score(hyps, [refs]).score,
                bleu.corpus_score(hyps, [refs]).score,
                len(common),
            )
        )
    rows.sort(key=lambda r: -r[1])

    width = max(len(r[0]) for r in rows)
    print("\n%-*s  %8s  %6s  %5s" % (width, "system", "chrF++", "BLEU", "n"))
    print("-" * (width + 26))
    for name, c, b, n in rows:
        print("%-*s  %8.2f  %6.2f  %5d" % (width, name, c, b, n))

    print("\nHow often each model is the best one for a sentence (by chrF++ against the gold):")
    for m in models:
        print("  %5.1f%%  %s" % (100.0 * oracle_picks[m] / len(common), m))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "chrF++", "BLEU", "n"])
        for name, c, b, n in rows:
            w.writerow([name, "%.2f" % c, "%.2f" % b, n])
    print("\nSaved: %s" % args.output)
    if len(common) < 100:
        print("Note: with fewer than 100 sentences these scores are noisy; treat small differences with care.")


if __name__ == "__main__":
    main()
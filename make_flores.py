"""
make_flores.py - prepare a public test set (such as FLORES-200) for the FAIR
evaluation.

Why: Bilal says Param-2 and IndicTrans2 were probably trained on the BPCC pairs
that make_pairs.py uses, so scores against those references are inflated for
those models. FLORES is meant for testing only, so it is the fairer check.

You give this script two plain text files with one sentence per line, in the
same order: the English sentences and their human Hindi translations (in
FLORES-200 these are the "eng_Latn" and "hin_Deva" files of the dev or devtest
split, about 1,000 lines each). Download them yourself (for example from the
"flores200" dataset on Hugging Face); this script does not download anything.

It writes into its own folder (default data_flores) so it never touches the
BPCC files:
  data_flores/source_sentences.txt   the English sentences
  data_flores/gold_all.csv           id, source, gold_reference

Then run the pipeline on that folder:
  python run_api_pipeline.py --input data_flores/source_sentences.txt ^
      --output data_flores/candidate_translations.csv --cache data_flores/api_cache.jsonl
  python evaluate.py data_flores/candidate_translations.csv data_flores/gold_all.csv ^
      --output data_flores/evaluation_results.csv

Usage:
    python make_flores.py english.txt hindi.txt
    python make_flores.py english.txt hindi.txt --n 500
"""

import argparse
import csv
import os
import re
import sys


def read_lines(path):
    if not os.path.exists(path):
        sys.exit("File not found: %s" % path)
    with open(path, encoding="utf-8-sig") as f:
        return [re.sub(r"\s+", " ", line).strip() for line in f.read().split("\n")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("english", help="text file, one English sentence per line")
    p.add_argument("hindi", help="text file, the Hindi translation of each line, same order")
    p.add_argument("--n", type=int, help="only use the first N lines")
    p.add_argument("--out-dir", default="data_flores")
    args = p.parse_args()

    en, hi = read_lines(args.english), read_lines(args.hindi)
    # A final empty line from the end of the file is normal; anything else is not.
    while en and not en[-1]:
        en.pop()
    while hi and not hi[-1]:
        hi.pop()
    if len(en) != len(hi):
        sys.exit(
            "The files have a different number of lines (English %d, Hindi %d), so the "
            "sentences cannot be matched up. Check that both come from the same split."
            % (len(en), len(hi))
        )

    rows, seen = [], set()
    for e, h in zip(en, hi):
        if e and h and e not in seen:
            seen.add(e)
            rows.append((e, h))
    skipped = len(en) - len(rows)
    if args.n:
        rows = rows[: args.n]
    if not rows:
        sys.exit("No sentence pairs found.")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "source_sentences.txt"), "w", encoding="utf-8", newline="\n") as f:
        for e, _ in rows:
            f.write(e + "\n")
    with open(os.path.join(args.out_dir, "gold_all.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "gold_reference"])
        for i, (e, h) in enumerate(rows, start=1):
            w.writerow([i, e, h])

    print("Wrote %d sentence pairs to %s" % (len(rows), args.out_dir))
    if skipped:
        print("(%d lines were skipped because they were empty or repeated.)" % skipped)


if __name__ == "__main__":
    main()
"""
make_gold_template.py - make the blank file where the gold (human) Hindi
references go.

It takes the first 200 sentences of data/source_sentences.txt. Because
make_sample.py shuffles its sample, these 200 are a random slice of your 10,000,
and a pilot run with --limit 200 translates exactly these sentences. The ids
match the ids in data/candidate_translations.csv.

Output: data/gold_template.csv with columns
        id, source, gold_reference   (gold_reference is empty - to be filled in)

Fill in gold_reference with a correct Hindi translation of each sentence (or
get it from a public English-Hindi test set if your guide allows that - then
this file just needs the id and gold_reference columns to match your sentences).
Open the CSV in VS Code or Excel and save it as UTF-8 so the Hindi stays intact.

The script refuses to overwrite an existing gold file unless you pass --force,
so work you have already filled in cannot be lost by accident.

Usage:
    python make_gold_template.py
    python make_gold_template.py --n 100
"""

import argparse
import csv
import os
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/source_sentences.txt")
    p.add_argument("--output", default="data/gold_template.csv")
    p.add_argument("--n", type=int, default=200, help="how many sentences (default 200)")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = p.parse_args()

    if not os.path.exists(args.input):
        sys.exit("Input not found: %s. Run make_sample.py first." % args.input)
    if os.path.exists(args.output) and not args.force:
        sys.exit(
            "%s already exists, so nothing was changed (it may hold references you have "
            "filled in). Use --force to overwrite it." % args.output
        )

    sentences = []
    seen = set()
    with open(args.input, encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if s and s not in seen:  # same rule as run_api_pipeline.py, so ids match
                seen.add(s)
                sentences.append(s)
    sentences = sentences[: args.n]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "gold_reference"])
        for i, s in enumerate(sentences, start=1):
            w.writerow([i, s, ""])

    print("Wrote %d sentences to %s (gold_reference is empty, to be filled in)." % (len(sentences), args.output))


if __name__ == "__main__":
    main()
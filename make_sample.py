"""
make_sample.py - pick a sample of clean English sentences from Bilal's
sentence_validation_results.csv for the translation pipeline.

What it does
  1. Reads only the columns it needs (the file is ~800 MB; the long "reasoning"
     and "context" columns are skipped to save memory).
  2. Keeps rows where is_cut_off is "no" (the sentence was not cut off).
  3. Uses the corrected_sentence column, drops empty and duplicate sentences.
  4. Keeps sentences of --min-words to --max-words words (default 6 to 60);
     very short or very long sentences translate poorly and are not useful
     for comparing models.
  5. Picks --n sentences (default 10000) SPREAD EVENLY ACROSS THE DICTIONARY
     WORDS: one sentence per word first, then a second sentence per word, and
     so on until --n is reached. Words with few sentences drop out early. If
     there are more words than --n, a random subset of words gets one each.
     (Use --no-even for a plain random sample instead.)
  6. Shuffles the final list with a fixed --seed, so the first 200 lines are
     still a random slice (used for the gold references) and everyone who runs
     this with the same file and settings gets the same sample.

Writes (in --out-dir, default "data"):
  source_sentences.txt   one sentence per line - the input of run_api_pipeline.py
  sample_meta.csv        id, row_id and word for each sentence, to trace it back
                         to Bilal's file

Usage:
  pip install pandas numpy
  python make_sample.py data/sentence_validation_results.csv

IMPORTANT: do not rerun this after you have started translating. It rebuilds
the sentence list, and the ids in your translations would no longer match.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

NEEDED = ["row_id", "word", "is_cut_off", "corrected_sentence"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_path", help="path to sentence_validation_results.csv")
    p.add_argument("--n", type=int, default=10000, help="how many sentences to pick (default 10000)")
    p.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    p.add_argument("--min-words", type=int, default=6)
    p.add_argument("--max-words", type=int, default=60)
    p.add_argument("--no-even", action="store_true", help="plain random sample instead of spreading evenly across words")
    p.add_argument("--out-dir", default="data", help="where to write the results (default: data)")
    args = p.parse_args()

    if not os.path.exists(args.csv_path):
        sys.exit("File not found: %s" % args.csv_path)

    # Check the header first, so a wrong file gives a clear message.
    try:
        header = list(pd.read_csv(args.csv_path, nrows=0, encoding="utf-8-sig").columns)
    except Exception as e:
        sys.exit("Could not read the file: %s" % e)
    missing = [c for c in NEEDED if c not in header]
    if missing:
        sys.exit(
            "These columns are missing from the file: %s. Found: %s"
            % (", ".join(missing), ", ".join(header))
        )

    # Read only the needed columns, as text. Quoted fields that contain line
    # breaks are handled correctly by the CSV parser.
    df = pd.read_csv(args.csv_path, usecols=NEEDED, dtype=str, encoding="utf-8-sig")

    print("Rows in file: %d" % len(df))

    flag = df["is_cut_off"].fillna("").str.strip().str.lower()
    print("is_cut_off values: %s" % flag.value_counts().to_dict())

    df = df[flag == "no"].copy()

    # One sentence per line in the output, so collapse any inner whitespace.
    df["sentence"] = (
        df["corrected_sentence"]
        .fillna("")
        .map(lambda s: re.sub(r"\s+", " ", s).strip())
    )
    df["word"] = df["word"].fillna("").str.strip()
    df = df[df["sentence"] != ""]
    df = df.drop_duplicates(subset="sentence")

    n_words = df["sentence"].str.split().str.len()
    df = df[(n_words >= args.min_words) & (n_words <= args.max_words)]

    print("Usable sentences after filtering: %d" % len(df))
    print("Distinct dictionary words among them: %d" % df["word"].nunique())
    if df.empty:
        sys.exit("Nothing left after filtering. Check the column values.")

    n = min(args.n, len(df))
    if n < args.n:
        print("Only %d usable sentences, so the sample is smaller than %d." % (n, args.n))

    if args.no_even:
        sample = df.sample(n=n, random_state=args.seed)
    else:
        # Shuffle, number the sentences within each word (0, 1, 2, ...), then
        # take the lowest numbers first: every word's 1st sentence, then every
        # word's 2nd, and so on. A random tie-break decides which words get
        # in when a round cannot be completed.
        rng = np.random.default_rng(args.seed)
        shuffled = df.sample(frac=1, random_state=args.seed).copy()
        shuffled["_rank"] = shuffled.groupby("word").cumcount()
        shuffled["_tie"] = rng.random(len(shuffled))
        sample = shuffled.sort_values(["_rank", "_tie"]).head(n)

    # Final shuffle, so the first lines (used for the gold subset) are random.
    sample = sample.sample(frac=1, random_state=args.seed + 1).reset_index(drop=True)

    per_word = sample.groupby("word").size()
    print(
        "Sample covers %d distinct words (%d to %d sentences per word)."
        % (len(per_word), per_word.min(), per_word.max())
    )

    os.makedirs(args.out_dir, exist_ok=True)
    txt_path = os.path.join(args.out_dir, "source_sentences.txt")
    meta_path = os.path.join(args.out_dir, "sample_meta.csv")

    with open(txt_path, "w", encoding="utf-8", newline="\n") as f:
        for s in sample["sentence"]:
            f.write(s + "\n")

    meta = pd.DataFrame(
        {
            "id": range(1, len(sample) + 1),
            "row_id": sample["row_id"],
            "word": sample["word"],
            "sentence": sample["sentence"],
        }
    )
    meta.to_csv(meta_path, index=False, encoding="utf-8-sig")

    print("Wrote %d sentences to %s" % (len(sample), txt_path))
    print("Wrote the trace-back table to %s" % meta_path)


if __name__ == "__main__":
    main()
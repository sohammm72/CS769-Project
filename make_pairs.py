"""
make_pairs.py - build the sentence list AND the gold Hindi references from
Bilal's pairs file (anneal_v2.jsonl).

Each line of that file is one JSON record, for example:
  {"messages": [{"role": "user", "content": "Translate the following English
   sentence to Hindi:\\n<English sentence>"}, {"role": "assistant", "content":
   "<translation>"}], "sl": "eng_Latn", "tl": "hin_Deva", "direction":
   "forward", "source": "bpcc_human:ilci", "synthetic": false}
The file covers many Indian languages (about 8.6 million lines, 4.7 GB).

What this script does
  1. Reads the file one line at a time (it never loads it all into memory) and
     keeps records that are English -> Hindi: tl = hin_Deva, sl = eng_Latn,
     direction = forward, and synthetic = false (human translations).
  2. Takes the English sentence from the user message (the instruction line is
     removed) and the Hindi from the assistant message.
  3. Keeps English sentences of --min-words to --max-words words (default 6 to
     60), drops empty pairs, and drops repeated English sentences.
  4. Picks --n pairs (default 10000) at random with a fixed --seed, or spread
     evenly across the datasets with --balance-sources, then shuffles them.
  5. Writes (in --out-dir, default "data"):
       source_sentences.txt   one English sentence per line  (input of run_api_pipeline.py)
       gold_all.csv           id, source, gold_reference     (input of evaluate.py)
       pairs_meta.csv         id, dataset the pair came from (for your own checks)
     Line N of source_sentences.txt is id N in both CSV files.

Uses only Python's built-in libraries, so it runs on the server as it is:
  python3 make_pairs.py /data1/bilalbaig/translation_research/anneal_v2.jsonl

Second, non-overlapping set (for example a test set next to the first set):
  python3 make_pairs.py /data1/bilalbaig/translation_research/anneal_v2.jsonl \\
      --exclude data/source_sentences.txt --out-dir data_test --seed 7

It only reads the pairs file. It refuses to overwrite existing outputs (use
--force), because rerunning would change the sentence list and the ids of any
translations you already made would no longer match.

CAUTION: the Hindi side is probably training data for some of the models being
compared (Bilal: "Param-2 / IndicTrans2 probably were trained on this"), so
scores against it can be inflated for those models. Use a held-out set such as
FLORES for the fair comparison.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

SL, TL, DIRECTION = "eng_Latn", "hin_Deva", "forward"


def extract_pair(rec):
    """Return (english, hindi) from one record, or None if it is not usable."""
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return None
    user = msgs[0].get("content") if isinstance(msgs[0], dict) else None
    asst = msgs[1].get("content") if isinstance(msgs[1], dict) else None
    if not isinstance(user, str) or not isinstance(asst, str):
        return None
    # The user message is an instruction line, a line break, then the sentence.
    if user.lstrip().lower().startswith("translate"):
        parts = user.split("\n", 1)
        if len(parts) < 2:
            return None
        user = parts[1]
    english = re.sub(r"\s+", " ", user).strip()
    hindi = re.sub(r"\s+", " ", asst).strip()
    if not english or not hindi:
        return None
    return english, hindi


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def read_excluded(paths):
    """English sentences to leave out: .txt (one per line) or .csv with a 'source' column."""
    excluded = set()
    for path in paths:
        if not os.path.exists(path):
            sys.exit("--exclude file not found: %s" % path)
        if path.lower().endswith(".csv"):
            with open(path, encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if "source" not in (reader.fieldnames or []):
                    sys.exit("--exclude CSV %s needs a 'source' column." % path)
                for row in reader:
                    excluded.add(norm(row["source"]))
        else:
            with open(path, encoding="utf-8-sig") as f:
                for line in f:
                    if line.strip():
                        excluded.add(norm(line))
    return excluded


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsonl_path", help="path to anneal_v2.jsonl")
    p.add_argument("--exclude", nargs="+", metavar="FILE", help="leave out every English sentence in these files (a .txt with one sentence per line, or a .csv with a 'source' column such as gold_all.csv). Use it to make a second set that does not overlap the first.")
    p.add_argument("--n", type=int, default=10000, help="how many pairs to pick (default 10000)")
    p.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    p.add_argument("--min-words", type=int, default=6)
    p.add_argument("--max-words", type=int, default=60)
    p.add_argument("--balance-sources", action="store_true", help="spread the sample evenly across the datasets (ilci, seed, massive...) instead of plain random")
    p.add_argument("--sources", nargs="+", help="only keep records whose 'source' starts with one of these (e.g. bpcc_human)")
    p.add_argument("--out-dir", default="data", help="where to write the results (default: data)")
    p.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = p.parse_args()

    if not os.path.exists(args.jsonl_path):
        sys.exit("File not found: %s" % args.jsonl_path)

    txt_path = os.path.join(args.out_dir, "source_sentences.txt")
    gold_path = os.path.join(args.out_dir, "gold_all.csv")
    meta_path = os.path.join(args.out_dir, "pairs_meta.csv")
    existing = [x for x in (txt_path, gold_path, meta_path) if os.path.exists(x)]
    if existing and not args.force:
        sys.exit(
            "These files already exist, so nothing was changed: %s\n"
            "Rerunning would change the sentence list, and the ids of any translations "
            "you already made would no longer match. Use --force only if you are sure."
            % ", ".join(existing)
        )

    excluded = read_excluded(args.exclude) if args.exclude else set()
    if excluded:
        print("Leaving out %d English sentences listed in --exclude." % len(excluded))

    scanned = 0
    hindi_target = 0
    bad_lines = 0
    drop = Counter()
    pool = {}  # english -> (hindi, dataset); first one seen wins
    dataset_counts = Counter()

    with open(args.jsonl_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            scanned += 1
            if scanned % 1000000 == 0:
                sys.stderr.write("  read %d lines...\n" % scanned)
            # Cheap test first, so the 8.6 million lines are not all parsed.
            if TL not in line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                bad_lines += 1
                continue
            if rec.get("tl") != TL:
                continue
            hindi_target += 1
            if rec.get("sl") != SL or rec.get("direction") != DIRECTION:
                drop["not English -> Hindi forward"] += 1
                continue
            if rec.get("synthetic") is not False:
                drop["synthetic or not marked as human"] += 1
                continue
            dataset = str(rec.get("source", ""))
            if args.sources and not any(dataset.startswith(s) for s in args.sources):
                drop["dataset not in --sources"] += 1
                continue
            pair = extract_pair(rec)
            if pair is None:
                drop["empty or unreadable messages"] += 1
                continue
            english, hindi = pair
            n_words = len(english.split())
            if n_words < args.min_words or n_words > args.max_words:
                drop["English length outside %d-%d words" % (args.min_words, args.max_words)] += 1
                continue
            if english in excluded:
                drop["already in the --exclude files"] += 1
                continue
            if english in pool:
                drop["repeated English sentence"] += 1
                continue
            pool[english] = (hindi, dataset)
            dataset_counts[dataset] += 1

    print("Lines read: %d (unreadable lines: %d)" % (scanned, bad_lines))
    print("Records with Hindi as target: %d" % hindi_target)
    for reason, count in drop.most_common():
        print("  dropped, %s: %d" % (reason, count))
    print("Usable English -> Hindi human pairs: %d" % len(pool))
    for dataset, count in dataset_counts.most_common():
        print("  %7d  %s" % (count, dataset))
    if not pool:
        sys.exit("No usable pairs found. Check the file path and the field values.")

    n = min(args.n, len(pool))
    if n < args.n:
        print("Only %d usable pairs, so the sample is smaller than %d." % (n, args.n))

    rng = random.Random(args.seed)
    items = sorted(pool.items())  # fixed order, so the same seed gives the same sample
    if args.balance_sources:
        by_dataset = defaultdict(list)
        for english, (hindi, dataset) in items:
            by_dataset[dataset].append((english, hindi, dataset))
        for lst in by_dataset.values():
            rng.shuffle(lst)
        chosen = []
        names = sorted(by_dataset)
        round_no = 0
        while len(chosen) < n:
            took = False
            batch = []
            for name in names:
                if round_no < len(by_dataset[name]):
                    batch.append(by_dataset[name][round_no])
                    took = True
            rng.shuffle(batch)  # random choice of which datasets fit in a last, partial round
            chosen.extend(batch)
            if not took:
                break
            round_no += 1
        chosen = chosen[:n]
    else:
        flat = [(english, hindi, dataset) for english, (hindi, dataset) in items]
        chosen = rng.sample(flat, n)
    rng.shuffle(chosen)  # final shuffle: the first lines are a random slice too
    if excluded:
        overlap = sum(1 for english, _, _ in chosen if english in excluded)
        assert overlap == 0, "overlap with --exclude files: %d" % overlap
        print("Overlap with the --exclude files: 0 sentences (checked).")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8", newline="\n") as f:
        for english, _, _ in chosen:
            f.write(english + "\n")
    with open(gold_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "gold_reference"])
        for i, (english, hindi, _) in enumerate(chosen, start=1):
            w.writerow([i, english, hindi])
    with open(meta_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "dataset"])
        for i, (_, _, dataset) in enumerate(chosen, start=1):
            w.writerow([i, dataset])

    mix = Counter(dataset for _, _, dataset in chosen)
    print("\nSample of %d pairs, by dataset:" % len(chosen))
    for dataset, count in mix.most_common():
        print("  %5d  %s" % (count, dataset))
    print("\nWrote %s" % txt_path)
    print("Wrote %s (the human Hindi references)" % gold_path)
    print("Wrote %s" % meta_path)
    print("Do not run this again once translating has started.")


if __name__ == "__main__":
    main()
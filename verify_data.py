r"""
verify_data.py - check that your data folder is byte-for-byte the shared gold set.

The gold set (data\source_sentences.txt and data\gold_all.csv) is NOT in GitHub.
Everyone copies it from the same place on Bilal's server. The fingerprints (SHA-256)
of those two files are in data_checksums.txt, which IS in GitHub, so you can prove
that your copy is the same as everyone else's.

  python verify_data.py            check your files against data_checksums.txt
  python verify_data.py --write    (Soham only) write data_checksums.txt from your files

If it says MISMATCH: do not use the files. Copy them again from the server (README,
Step 8). Do not open gold_all.csv in Excel and save it - that changes the file.
"""
import argparse
import hashlib
import os
import sys

FILES = ["data/source_sentences.txt", "data/gold_all.csv"]
CHECKSUM_FILE = "data_checksums.txt"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def count_lines(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--write", action="store_true", help="write %s from the files you have" % CHECKSUM_FILE)
    args = p.parse_args()

    missing = [x for x in FILES if not os.path.exists(x)]
    if missing:
        sys.exit("Missing: %s\nCopy them from the server first (README, Step 8)." % ", ".join(missing))

    if args.write:
        with open(CHECKSUM_FILE, "w", encoding="utf-8", newline="\n") as f:
            for x in FILES:
                f.write("%s  %s\n" % (sha256(x), x))
        print("Wrote %s:" % CHECKSUM_FILE)
        print(open(CHECKSUM_FILE, encoding="utf-8").read())
        return

    if not os.path.exists(CHECKSUM_FILE):
        sys.exit("%s not found. Run `git pull` to get it." % CHECKSUM_FILE)
    expected = {}
    for line in open(CHECKSUM_FILE, encoding="utf-8"):
        parts = line.split()
        if len(parts) == 2:
            expected[parts[1]] = parts[0]

    bad = 0
    for x in FILES:
        got = sha256(x)
        ok = expected.get(x) == got
        bad += 0 if ok else 1
        print("%-28s %s  (%d lines)" % (x, "OK" if ok else "MISMATCH", count_lines(x)))
    if bad:
        sys.exit("\nYour data does not match the shared gold set. Copy the files from the server again.")
    print("\nAll good: you have exactly the shared gold set.")


if __name__ == "__main__":
    main()  
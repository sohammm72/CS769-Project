"""
run_api_pipeline.py - send English sentences to the models on Bilal's vLLM
server and collect every model's Hindi translation in one CSV.

Because the server is only launched on request and may drop at any time:
  * every finished translation is saved at once to a cache file
    (data/api_cache.jsonl) - this is the permanent record, do not delete it;
  * if you rerun the same command, finished work is skipped and the run
    carries on where it stopped;
  * the server is checked first, and the script stops with a clear message
    if it is not launched, instead of failing on thousands of sentences;
  * after several failures in a row the run stops by itself (progress is kept);
  * a model that is listed but cannot answer (for example wrong request
    format) is flagged and skipped, the other models carry on.

Setup (once per terminal window) - see the top of api_client.py:
    set VLLM_BASE_URL=...      (Command Prompt; use $env:... in PowerShell)
    set VLLM_API_KEY=...

Usage:
    python run_api_pipeline.py --list-models
    python run_api_pipeline.py --models MODEL_ID --limit 5              (tiny test)
    python run_api_pipeline.py --models ID1 ID2 ID3 --limit 200         (pilot)
    python run_api_pipeline.py                                          (full run, all served models)

Input:  data/source_sentences.txt by default (made by make_sample.py), or any
        .txt file (one sentence per line) or .csv file (use --column).
Output: data/candidate_translations.csv (columns: id, source, one column per
        model named in this command). Each run rewrites this CSV with only
        the models named in that command; the cache keeps everything, so to
        rebuild a full CSV just name all the models again - nothing is
        translated twice.
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from api_client import (
    ConfigError,
    ServerRejected,
    ServerUnavailable,
    VLLMClient,
)

MAX_PRINTED_ERRORS_PER_MODEL = 3
MAX_REJECTIONS_IN_ROW = 5


# ---------------------------------------------------------------------- #
# Reading and writing files
# ---------------------------------------------------------------------- #
def read_sentences(path, column):
    """Read source sentences. Blank lines are skipped, duplicates collapsed."""
    if not os.path.exists(path):
        sys.exit("Input file not found: %s" % path)

    sentences = []
    if path.lower().endswith(".csv"):
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or column not in reader.fieldnames:
                sys.exit(
                    "Column '%s' not found in %s. Columns are: %s. "
                    "Use --column to pick one."
                    % (column, path, ", ".join(reader.fieldnames or []))
                )
            for row in reader:
                sentences.append((row[column] or "").strip())
    else:
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                sentences.append(line.strip())

    seen = set()
    unique = []
    for s in sentences:
        if s and s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def load_cache(path):
    """Return {(model, source): translation} from the cache file."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[(rec["model"], rec["source"])] = rec["translation"]
            except (ValueError, KeyError):
                # A half-written last line after a crash: ignore it, that
                # sentence will simply be translated again.
                continue
    return done


def write_csv(path, sentences, models, done):
    """Rebuild the output CSV from everything finished so far."""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "source"] + list(models))
        for i, src in enumerate(sentences, start=1):
            writer.writerow([i, src] + [done.get((m, src), "") for m in models])


# ---------------------------------------------------------------------- #
# Main run
# ---------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list-models", action="store_true", help="print the models the server serves and exit")
    p.add_argument("--input", default="data/source_sentences.txt", help=".txt (one sentence per line) or .csv file (default: data/source_sentences.txt, made by make_sample.py)")
    p.add_argument("--column", default="english", help="CSV column holding the English sentence (default: english)")
    p.add_argument("--output", default="data/candidate_translations.csv", help="output CSV (default: data/candidate_translations.csv)")
    p.add_argument("--cache", default="data/api_cache.jsonl", help="cache of finished translations (default: data/api_cache.jsonl)")
    p.add_argument("--models", nargs="+", help="model ids to use (default: every model the server serves)")
    p.add_argument("--limit", type=int, help="only use the first N sentences (for tests and the pilot)")
    p.add_argument("--src-lang", default="English")
    p.add_argument("--tgt-lang", default="Hindi")
    p.add_argument("--workers", type=int, default=8, help="parallel requests (default 8; lower it if the server struggles)")
    p.add_argument("--max-failures", type=int, default=10, help="stop after this many failures in a row (default 10)")
    p.add_argument("--timeout", type=int, default=120, help="seconds to wait for one translation (default 120)")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        client = VLLMClient.from_env(timeout=args.timeout)
    except ConfigError as e:
        print(e)
        return 2

    ok, message = client.check_server()
    if not ok:
        print("\nCannot start: " + message)
        print("Nothing was translated. Fix the problem above, then rerun this command.")
        return 2

    served = client.list_models()
    if args.list_models:
        print("Models served right now:")
        for m in served:
            print("  " + m)
        return 0

    models = args.models or served
    missing = [m for m in models if m not in served]
    if missing:
        print("These models are not served right now: " + ", ".join(missing))
        print("Served models: " + ", ".join(served))
        print("Copy the names exactly from --list-models, or ask Bilal to launch the missing ones.")
        return 2

    sentences = read_sentences(args.input, args.column)
    if args.limit:
        sentences = sentences[: args.limit]
    if not sentences:
        print("No sentences found in the input file.")
        return 2

    for path in (args.output, args.cache):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    done = load_cache(args.cache)
    todo = [(m, s) for m in models for s in sentences if (m, s) not in done]
    total = len(models) * len(sentences)
    print(
        "%d sentences x %d models = %d translations. Already done: %d. To do: %d."
        % (len(sentences), len(models), total, total - len(todo), len(todo))
    )

    state = {"failures_in_row": 0, "completed": 0, "stop_reason": None}
    per_model = {
        m: {"done": 0, "failed": 0, "rejected_in_row": 0, "first": None, "last": None, "errors": 0}
        for m in models
    }
    bad_models = {}  # model -> reason
    lock = threading.Lock()
    stop = threading.Event()
    cache_file = open(args.cache, "a", encoding="utf-8")

    def note_error(model, text):
        """Print only the first few errors of each model, to keep the screen readable."""
        info = per_model[model]
        info["failed"] += 1
        info["errors"] += 1
        if info["errors"] <= MAX_PRINTED_ERRORS_PER_MODEL:
            print("  [%s] %s" % (model, text))

    def work(model, source):
        if stop.is_set() or model in bad_models:
            return
        with lock:
            if per_model[model]["first"] is None:
                per_model[model]["first"] = time.time()
        try:
            text = client.translate(model, source, args.src_lang, args.tgt_lang)
        except ServerRejected as e:
            with lock:
                note_error(model, str(e))
                reason = None
                if e.status_code in (401, 403):
                    state["stop_reason"] = str(e)
                    stop.set()
                elif e.status_code == 404:
                    reason = "the server does not know this model (HTTP 404)"
                else:
                    per_model[model]["rejected_in_row"] += 1
                    if per_model[model]["rejected_in_row"] >= MAX_REJECTIONS_IN_ROW:
                        reason = "it refused %d requests in a row" % MAX_REJECTIONS_IN_ROW
                if reason and model not in bad_models:  # print once, even with parallel workers
                    bad_models[model] = reason
                    print("  Skipping model %s from now on: %s." % (model, reason))
            return
        except ServerUnavailable as e:
            with lock:
                note_error(model, str(e))
                state["failures_in_row"] += 1
                if state["failures_in_row"] >= args.max_failures:
                    state["stop_reason"] = str(e)
                    stop.set()
            return

        with lock:
            state["failures_in_row"] = 0
            per_model[model]["rejected_in_row"] = 0
            if not text:
                note_error(model, "empty answer for one sentence (left for a rerun)")
                return
            done[(model, source)] = text
            cache_file.write(
                json.dumps(
                    {"model": model, "source": source, "translation": text},
                    ensure_ascii=False,
                )
                + "\n"
            )
            cache_file.flush()
            per_model[model]["done"] += 1
            per_model[model]["last"] = time.time()
            state["completed"] += 1
            if state["completed"] % 25 == 0 or state["completed"] == len(todo):
                print("  %d / %d done this run" % (state["completed"], len(todo)))

    executor = ThreadPoolExecutor(max_workers=max(1, args.workers))
    try:
        futures = [executor.submit(work, m, s) for m, s in todo]
        for fut in futures:
            fut.result()
    except KeyboardInterrupt:
        stop.set()
        print("\nStopping... progress so far is saved.")
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        cache_file.close()
        write_csv(args.output, sentences, models, done)

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\nThis run, per model:")
    for m in models:
        info = per_model[m]
        if info["done"] and info["first"] and info["last"]:
            secs = max(info["last"] - info["first"], 0.001)
            speed = "%d translated in %.0f s (%.1f per second)" % (info["done"], secs, info["done"] / secs)
        else:
            speed = "0 translated"
        extra = ""
        if info["failed"]:
            extra += ", %d failed" % info["failed"]
        if m in bad_models:
            extra += ", SKIPPED (%s)" % bad_models[m]
        print("  %s: %s%s" % (m, speed, extra))
    print("(Per-model times overlap when several models run at once. Use them as a rough guide only.)")

    remaining = {m: sum(1 for s in sentences if (m, s) not in done) for m in models}
    remaining_total = sum(remaining.values())
    print("\nSaved: %s" % args.output)
    if state["stop_reason"]:
        print("Run stopped early: " + state["stop_reason"])
    if remaining_total == 0:
        print("All %d translations are complete." % total)
        return 0
    for m, r in remaining.items():
        if r:
            print("  still missing for %s: %d" % (m, r))
    print("Rerun the same command to continue" + (" (models marked SKIPPED need Bilal's help first)." if bad_models else "."))
    return 1


if __name__ == "__main__":
    sys.exit(main())
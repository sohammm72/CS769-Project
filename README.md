# CS 769 Project: Multi-Model English→Hindi Translation Combiner

Given an English sentence, we get Hindi translations from several machine
translation (MT) models and combine them into one better translation: majority
voting (consensus), quality estimation (QE) reranking, and QE-fusion from the paper
"Don't Rank, Combine!" (Vernikos & Popescu-Belis, 2024, arXiv 2401.06688).
The combined output can then serve as synthetic training data.

Guide: Prof. Venkatapathy S. Mentor: Bilal. Team: Soham Murudkar, Sanit Jethwani.

**This README takes a new or existing member from zero to where the project is now,
and lists what is left.** Follow the steps in order. Every error we hit on the way
is in the Troubleshooting table, so check it before asking.

---

## Where we are (read this first)

| # | Stage | Status |
|---|---|---|
| 1 | 5 MT models running on your own laptop (Steps 1-7) | Done |
| 2 | Shared gold set: 10,000 English→Hindi human pairs (Step 8) | Done |
| 3 | Your own test set, different from the gold set (Step 9) | Done per person |
| 4 | Translate with the models on Bilal's server (Step 10) | Waiting for Bilal to launch the models |
| 5 | Consensus baseline and evaluation (Steps 12-13) | Scripts done |
| 6 | Real QE (CometKiwi) scoring and QE-reranking (Step 11) | Done and tested on the laptop (CPU) |
| 7 | QE-fusion (the paper's Algorithm 1) | Remaining, in progress |
| 8 | Everything else (see **Roadmap**) | Remaining |

## Rules (please read)

1. **Never put a password, token or key in a file, in code, in a commit or in a chat.**
   That means Bilal's server password, the vLLM API key and your Hugging Face token.
2. **Data stays out of GitHub.** `data/`, `data_test/` and any `data_*/` folder come from
   Bilal's training-pairs file. They are in `.gitignore`. Check `git status` before every push.
3. **On Bilal's server, only read and copy.** Work inside `/data1/bilalbaig/soham/`. There are
   no GPUs on the server. The server has no internet and is reachable only from the IIT Bombay
   network or VPN.
4. **Never rerun `make_pairs.py` for the gold set** (it refuses to overwrite, so never use `--force`).
   A rerun would change the sentences and the ids of every translation already made.
5. **Do not tune anything on a test set.** Tune on `data/`, report on a test set.
6. **The `demo` scorer in `qe_score.py` is not real QE.** It exists only to test the code.
   Never report its numbers. Real QE means `--scorer cometkiwi`.
7. **Caution on scores:** Bilal says Param-2 and IndicTrans2 were probably trained on these pairs,
   so their scores on `data/` can be inflated. A held-out set (FLORES) is the fair comparison.

## What you need before starting

- Windows, VS Code, and **Command Prompt** as the terminal (all commands below are for cmd, not PowerShell)
- Python 3.10 or newer for Phase 1, **plus Python 3.12 for the QE environment (Step 11)**.
  Check what you have with `py -0`. Python 3.12 is at https://www.python.org/downloads/
- About **20 GB free disk space** (models 10-15 GB, QE model 2.3 GB)
- A good internet connection (the downloads are large)
- A free Hugging Face account (needed for IndicTrans2 and for CometKiwi)
- Access to Bilal's server (IITB network or VPN, and the password from Soham or Bilal)
- An NVIDIA GPU is optional. Everything works on CPU, just slower
- Git or GitHub Desktop

Commands that start with `set` are for Command Prompt. If you are in PowerShell, `set X=1`
and `$env:X` will not do what you expect. Open a Command Prompt terminal (Step 1).

---

# PHASE 1: the 5 models on your laptop

| Key | Model | Hugging Face ID | Login needed? |
|---|---|---|---|
| `opus` | opus-mt-en-hi | `Helsinki-NLP/opus-mt-en-hi` | No |
| `nllb` | NLLB-200 (600M) | `facebook/nllb-200-distilled-600M` | No |
| `mbart50` | mBART-50 | `facebook/mbart-large-50-one-to-many-mmt` | No |
| `indictrans2` | IndicTrans2 (200M) | `ai4bharat/indictrans2-en-indic-dist-200M` | **Yes (gated)** |
| `qwen` | Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | No |

## Step 1: Get the project and open a Command Prompt terminal

1. Clone this repository (in GitHub Desktop: File → Clone repository).
2. In VS Code: File → Open Folder, and choose the cloned folder.
3. Open a terminal: Terminal → New Terminal.
4. Click the dropdown arrow next to the `+` in the terminal panel and choose **Command Prompt**.
   The prompt should look like `C:\Users\you\CS769 Project>`.

All commands below are run in this terminal, from the project folder.

## Step 2: Create and activate a virtual environment

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

Your prompt must now start with `(venv)`. **Every time you open a new terminal,
run `venv\Scripts\activate` again.**

> `source venv/bin/activate` is a Linux/Mac command and does not work on Windows.

Then tell VS Code to use this environment: press `Ctrl+Shift+P`, choose
**Python: Select Interpreter**, and pick the one at `.\venv\Scripts\python.exe`.

## Step 3: Install PyTorch (do this BEFORE anything else)

First check whether you have an NVIDIA GPU:

```bat
nvidia-smi
```

- If it prints a table, you have a GPU. Note the **CUDA Version** in the top right.
- If it says "not recognized", you have no NVIDIA GPU. Use the CPU install.

**No GPU (CPU only):**
```bat
pip install torch
```

**With an NVIDIA GPU:** a plain `pip install torch` on Windows installs the
**CPU-only** build, which is much slower. Instead, go to https://pytorch.org,
choose Stable, Windows, Pip, Python, and the CUDA version that matches (or is
lower than) the one from `nvidia-smi`, and copy the command it gives you. It
looks like:

```bat
pip install torch --index-url https://download.pytorch.org/whl/cu126
```
(the `cu126` part depends on your CUDA version)

**Verify:**
```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- GPU users: you want `True` and a version ending like `+cu126`.
- If you have a GPU but see `False` or a version ending in `+cpu`, run
  `pip uninstall -y torch torchvision torchaudio`, then repeat the GPU install above.
  Also update your NVIDIA driver and open a new terminal.
- CPU-only users: `False` is expected.

## Step 4: Install the other libraries

```bat
pip install -r requirements.txt
```

This installs `transformers` version 4.x on purpose. **Do not upgrade
`transformers` to 5.x**, because IndicTransToolkit breaks with it.
`requirements.txt` also includes `requests`, `pandas`, `numpy` and `sacrebleu`, which the pipeline scripts in Steps 10-13 use.

### IndicTransToolkit (needs a C++ compiler on Windows)

IndicTrans2 needs the `IndicTransToolkit` package, which has to be compiled.
On Windows that requires the Microsoft C++ Build Tools.

1. Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run the installer and tick **Desktop development with C++**, then Install.
   (This is several GB and takes a while.)
3. When it finishes, just close the installer. You don't need to launch anything.
4. **Close the VS Code terminal and open a new one** (Command Prompt), then:

```bat
venv\Scripts\activate
pip install IndicTransToolkit
```

Check it:
```bat
python -c "from IndicTransToolkit.processor import IndicProcessor; print('ok')"
```
It should print `ok`.

If you see `error: Microsoft Visual C++ 14.0 or greater is required`, the C++
workload wasn't installed or the terminal wasn't restarted. Open "Visual Studio
Installer" from the Start menu, click **Modify** on Build Tools, make sure
**Desktop development with C++** is ticked, apply, then retry in a fresh terminal.

If you see `ImportError: cannot import name 'PreTrainedTokenizerBase'`, you have
transformers 5.x. Fix it with `pip install "transformers<5"`.

## Step 5: Hugging Face login (required for IndicTrans2, and later for CometKiwi)

IndicTrans2 is a **gated model**: you must accept its terms and log in, or the
download fails with `401 ... Access to model ... is restricted`.
Each person needs to do this with their own account.

1. Create a free account at https://huggingface.co and verify your email.
2. While logged in, open https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M
   and submit the access form. Approval is usually quick. Once it is approved, the
   page shows the normal model files instead of the form.
3. Create a token at https://huggingface.co/settings/tokens
   (**Create new token**, type **Read**). Copy it. It starts with `hf_`.
   **Never share it, paste it into code, or commit it to GitHub.**
4. Log in from the terminal:
   ```bat
   hf auth login
   ```
   (If `hf` isn't recognized, use `huggingface-cli login`.)
   Paste the token when asked. Nothing shows on screen as you paste, which is
   normal. If asked "Add token as git credential?", answer `n`.
5. Check:
   ```bat
   hf auth whoami
   ```

You will do the same "accept the terms" step for CometKiwi in Step 11. The token is shared by
all models, but **every gated model needs its own "accept" click on its page.**

## Step 6: Download the models into the project folder

```bat
python download_models.py
```

This downloads all 5 models (about 10-15 GB) into a `models\` folder in the
project. It can take a long time. If it stops or your internet drops, run the
same command again and it resumes where it left off.

When done you should have:
```
models\opus
models\nllb
models\mbart50
models\indictrans2
models\qwen
```

The `models\` folder is **not stored in GitHub** (too big). Each person
downloads their own copy with this step.

## Step 7: Test the models

Test one model at a time (smallest first):

```bat
python test_models.py opus
python test_models.py nllb
python test_models.py mbart50
python test_models.py indictrans2
python test_models.py qwen
```

Or all five in one go:

```bat
python test_models.py
```

The Hindi output is written to **`outputs\test_results.txt`**. Open that file in
VS Code to read it. Command Prompt often cannot display Hindi (you get boxes or
an encoding error), but VS Code shows it fine.

A model that fails prints `[name] FAILED: ...` and the script carries on with the
others. To test your own sentences, edit the `SENTENCES` list at the top of `test_models.py`.

You're done with Phase 1 when all five models give sensible Hindi sentences.
(`outputs\test_results.txt` is also the input for the QE test in Step 11.)

---

# PHASE 2: the data

We use two kinds of sets, and they must never overlap:

| Folder | What it is | Same for everyone? |
|---|---|---|
| `data\` | **The gold set**: 10,000 English sentences with human Hindi references. We build and tune the combiners on it. | **Yes, exactly the same files** |
| `data_test\` | **Your test set**: 10,000 other pairs, none of them in `data\`. Used only for the final numbers. | No, each person makes their own |

Both come from Bilal's pairs file `/data1/bilalbaig/translation_research/anneal_v2.jsonl`
(4.7 GB, many languages). `make_pairs.py` keeps the English→Hindi human pairs.
The gold set was already built once, with the default seed 42, on the server in
`/data1/bilalbaig/soham/data/`. **Do not rebuild it. Copy it.**

## Step 8: Copy the gold set from the server (Sanit and anyone new)

You need the IITB network or VPN, and the server password (ask Soham or Bilal; never write it down in a file).
Work only in `/data1/bilalbaig/soham/`. Do not create your own folder on the server for the gold
set, because that would mean redoing the whole process.

First look at what is there:
```bat
ssh bilal@10.129.6.181
ls /data1/bilalbaig/soham/data
exit
```
The first time, it asks "Are you sure you want to continue connecting?" Type `yes`.
You should see `source_sentences.txt`, `gold_all.csv` and `pairs_meta.csv`.

Back on your laptop, in the project folder (the one containing `make_pairs.py`):
```bat
mkdir data
scp bilal@10.129.6.181:/data1/bilalbaig/soham/data/source_sentences.txt data\
scp bilal@10.129.6.181:/data1/bilalbaig/soham/data/gold_all.csv data\
```

Now check you got exactly the shared files (compares fingerprints with `data_checksums.txt` from GitHub):
```bat
python verify_data.py
```
It must print `OK` for both files and `All good`. If it says `MISMATCH`, copy the files again.
Do not open `gold_all.csv` in Excel and save it; that changes the file. Open it in VS Code instead.

Line N of `source_sentences.txt` is `id` N in `gold_all.csv` (columns `id, source, gold_reference`).

## Step 9: Make your own test set (different from the gold set)

This is done **on the server**, because only the server has the 4.7 GB pairs file. The script uses only
built-in Python, so no installs. The `--exclude` option guarantees that no sentence of the gold set appears in your test set.
Use your own seed (any number except 42; Soham used 7, Sanit please use 11).

```bat
ssh bilal@10.129.6.181
cd /data1/bilalbaig/soham
python3 make_pairs.py /data1/bilalbaig/translation_research/anneal_v2.jsonl --exclude data/source_sentences.txt --out-dir data_test_sanit --seed 11
exit
```
It takes a few minutes. At the end it must print `Overlap with the --exclude files: 0 sentences (checked).`
(Soham's test set is in `data_test`; yours goes in `data_test_sanit`, so nothing is overwritten.)

Copy it to your laptop and give the files the names the scripts expect:
```bat
mkdir data_test
scp bilal@10.129.6.181:/data1/bilalbaig/soham/data_test_sanit/source_sentences.txt data_test\test_sentences.txt
scp bilal@10.129.6.181:/data1/bilalbaig/soham/data_test_sanit/gold_all.csv data_test\test_gold_all.csv
```
Your test set and Soham's are different random draws, so a score on one is not exactly comparable to a
score on the other. When we compare methods, each method is judged on the same set.

If `scp` says `Connection closed`, run it again. If it keeps failing, try `scp -O ...` or check that
`ssh` still works (the server may be busy or you may be off the IITB network).

---

# PHASE 3: translate with the models on Bilal's server

The server runs the models through a vLLM web API. Bilal launches them on request. Nothing here runs
until he says the models are up. **Ask him in the group chat, and ask for: the server address, the API key,
and which model IDs are being served.**

## Step 10: Run the translation pipeline

Set the address and key in the terminal window (they last only for that window).
Command Prompt:
```bat
set VLLM_BASE_URL=https://the-address-Bilal-gives-you
set VLLM_API_KEY=the-key
```
PowerShell uses `$env:VLLM_BASE_URL="..."` instead.
Never write the address, key or Bilal's password into a file, and never commit them.

```bat
python run_api_pipeline.py --list-models                       # which models are up, exact names
python run_api_pipeline.py --models MODEL_ID --limit 5         # tiny test: read the Hindi
python run_api_pipeline.py --models ID1 ID2 ID3 --limit 200    # pilot, smaller models first
python run_api_pipeline.py                                     # full run, all served models
```
- Everything finished is saved at once in `data\api_cache.jsonl` (the permanent record).
- If the server drops, rerun the same command later; nothing is translated twice.
- Output: `data\candidate_translations.csv` (one column per model; only the models named in that command;
  name them all again to rebuild the full CSV from the cache).
- `--workers N` changes parallel requests (default 8). A model the server lists but cannot answer is skipped and reported.
- Note when you start and finish each model: that tells you how much server time the full run needs.
- NLLB and IndicTrans2 are not chat models (they take language codes, not an instruction), so they
  cannot be called through this client until Bilal says how they are served.
- To translate your test set, add `--input data_test\test_sentences.txt --output data_test\candidate_translations.csv --cache data_test\api_cache.jsonl`.
- Back up `data\api_cache.jsonl` somewhere other than your laptop (not on GitHub).

If `--list-models` says `502`: the server or the tunnel is down. It is not your code. Tell Bilal.
(The old public Cloudflare tunnel died once; the server is normally reached from the IITB network or VPN.)

---

# PHASE 4: quality estimation (QE) and combining

## Step 11: Real QE with CometKiwi (separate environment, Python 3.12)

CometKiwi (`Unbabel/wmt22-cometkiwi-da`) looks at a source sentence and a translation, **without a reference**,
and gives a score between 0 and 1 (higher = better). This is the "QE" in QE-reranking and QE-fusion.

**It needs its own environment**, because the `unbabel-comet` package requires `numpy<2` and breaks on Python 3.13.
Do not install it into `venv`.

```bat
deactivate
py -3.12 -m venv venv_qe
venv_qe\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements_qe.txt
```
(`py -0` lists your Python versions. If 3.12 is missing, install it first. Your prompt must start with `(venv_qe)`.)

Accept the model terms **with the same Hugging Face account as your token**: open
https://huggingface.co/Unbabel/wmt22-cometkiwi-da, click to accept the license, wait for it to show the files.
Then log in in this environment (`hf auth login`, see Step 5).

Test on the 3 sentences from Step 7 (first run downloads about 2.3 GB; run it on its own and wait for it to finish):
```bat
python qe_score.py outputs\test_results.txt --scorer cometkiwi --out-scores outputs\real_qe_scores.csv --out-pick outputs\real_qe_rerank.csv
```
You should see "Predicting DataLoader 0: 100%" and a table like this (your numbers may differ in the 2nd or 3rd decimal):

```
  avg 0.642   opus
  avg 0.847   nllb
  avg 0.829   mbart50
  avg 0.862   indictrans2   (top-scored on 100%)
  avg 0.585   qwen
```
"GPU available: False, used: False" is normal and fine here. For 15 pairs the CPU takes about a second.
Using the GPU is optional and only matters at the 10,000-sentence scale: install a CUDA build of torch into `venv_qe`
(Step 3 command) and add `--gpus 1`. The bigger `wmt23-cometkiwi-da-xxl` does not fit a laptop; it will run on Bilal's cluster.

On the real candidates (after Step 10):
```bat
python qe_score.py data\candidate_translations.csv --scorer cometkiwi --out-scores data\qe_scores.csv --out-pick data\qe_rerank.csv
```
- It keeps a cache (`data\qe_cache.jsonl`), so a rerun does not score anything twice. Delete it if you change the model.
- `--batch-size 16` is the default. `--model` accepts a local `.ckpt` path (for offline use on the cluster).
- `data\qe_rerank.csv` is QE-reranking: for each sentence, the candidate with the top QE score.

## Step 12: Consensus baseline (no server, no QE needed)

For each sentence: if most models give the identical sentence, take it. Otherwise take the translation closest to all the others (chrF++ consensus).
```bat
python combine_baseline.py data\candidate_translations.csv
python combine_baseline.py data\candidate_translations.csv --processes 4    # faster on the full run
```
Output: `data\baseline_consensus.csv`.

## Step 13: Evaluate

```bat
python evaluate.py data\candidate_translations.csv data\gold_all.csv
python evaluate.py data\candidate_translations.csv data\gold_all.csv --extra QE-rerank=data\qe_rerank.csv --extra Consensus=data\baseline_consensus.csv
```
It prints chrF++ and BLEU for each model, for the baseline and for an **oracle** (the best possible pick per sentence; an upper bound).
Only sentences that every system has translated are scored (a pilot of 200 scores 200).
`--extra NAME=FILE` adds any method (a CSV with `id` and `translation`), so QE-fusion goes in the same way.

---

## Roadmap: what is left

Owners are decided in the group. Do these roughly in this order.

| # | Task | Notes |
|---|---|---|
| 1 | Full translation run on the gold set and the test set (Step 10) | Needs Bilal to launch the models. Pilot with 200 sentences first. |
| 2 | QE-reranking and consensus on the full set; evaluate (Steps 11-13) | Scripts exist. Compare each model, consensus, QE-rerank, oracle. |
| 3 | **QE-fusion** (`qe_fusion.py`) | Algorithm 1 of the paper: take the top-QE candidate as the base, find where the others differ (difflib), try replacing those spans, keep what raises the QE score (beam search). Needs the real CometKiwi. Will be added to this repo. |
| 4 | Final comparison on the held-out test set(s) | Models vs consensus vs QE-rerank vs QE-fusion vs oracle. Report FLORES too (fair comparison, see rule 7). |
| 5 | **Literature review: gold datasets for Indian languages** | IN22, FLORES-200, NTREX-128, and others: size, languages, licence, whether models trained on them. Written up for the guide. |
| 6 | **Pipeline for all Indian languages** (not only Hindi) | Needs a language table (codes for each model and for the QE scorer). `make_pairs.py` already reads `sl`/`tl`; it only needs those parameterised. |
| 7 | **Plug-and-play models** | One simple way to add any model (a local HF model, a vLLM model or the company's in-house model) and run the same pipeline. |
| 8 | CometKiwi XXL on Bilal's cluster | Check with Bilal whether the weights can be on the server (it has no internet), and how to submit jobs. |
| 9 | Pilot run with Bilal | Agree when it starts and what he needs from us. |
| 10 | Hand-off package for Bilal | Code plus a short guide he can run on his model and the company's model. |

## Project structure

```
CS769 Project\
  README.md               this file
  requirements.txt        packages for Phase 1 (venv)
  requirements_qe.txt     packages for QE (venv_qe, Python 3.12)
  .gitignore
  data_checksums.txt      fingerprints of the shared gold set
  verify_data.py          checks your data\ against data_checksums.txt
  download_models.py      Phase 1: downloads the 5 models into models\
  mt_models.py            Phase 1: loads each model and exposes a translate function
  test_models.py          Phase 1: runs the models on sample sentences
  make_pairs.py           builds a sentence list and human references from Bilal's pairs file (run on the server)
  api_client.py           talks to the vLLM server (list models, health check, translate, retries)
  run_api_pipeline.py     translates a sentence list with the server's models, resumable
  combine_baseline.py     consensus baseline
  qe_score.py             CometKiwi QE scores and QE-reranking
  evaluate.py             chrF++ / BLEU for models, baseline, oracle and extra methods
  make_flores.py          builds a FLORES test set from two line-aligned text files
  data\  data_test\       (not in GitHub) gold set and your test set
  models\ outputs\ venv\ venv_qe\    (not in GitHub)
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `'source' is not recognized` | Wrong OS command. Use `venv\Scripts\activate` |
| `'python' is not recognized` | Try `py` instead, or reinstall Python with "Add to PATH" ticked |
| `$env:...` gives an error in the terminal | That is PowerShell syntax. In Command Prompt use `set NAME=value` |
| `ModuleNotFoundError` | The venv is not active, or it is the wrong one. Phase 1 and Steps 10, 12, 13 use `(venv)`. QE (Step 11) uses `(venv_qe)`. Check the start of your prompt |
| `No module named 'comet'` | You are in `(venv)`, not `(venv_qe)` |
| `torch.cuda.is_available()` is `False` but you have an NVIDIA GPU | CPU-only torch was installed. Redo Step 3 (GPU install) |
| `Microsoft Visual C++ 14.0 or greater is required` | Install C++ Build Tools (Step 4), then open a NEW terminal |
| `cannot import name 'PreTrainedTokenizerBase'` | `pip install "transformers<5"` |
| `401 ... Access to model ... is restricted` | Complete Step 5 (access request, token, login) |
| `GatedRepoError` / `403` for CometKiwi | You did not accept the license on https://huggingface.co/Unbabel/wmt22-cometkiwi-da with the account your token belongs to. Accept it, wait until the page shows files, and check the token's permission |
| COMET says "Model not supported" or hides the real download error | `qe_score.py` downloads the model first so you see the real error. It is almost always the gated-access issue above |
| `numpy` `OverflowError` / `numpy.core` errors when importing comet | You are on Python 3.13 or have numpy 2.x. Make `venv_qe` with Python 3.12 (Step 11) |
| `No module named 'pkg_resources'` | `pip install setuptools==69.5.1` |
| QE run closes with no message, `echo %errorlevel%` shows `-1073741819` | Rerun with `python -X faulthandler qe_score.py ...` to see why. It worked on rerun for Soham. Make sure `qe_score.py` is the latest version (it uses `num_workers=0`) |
| "num_workers > 0 ... scores of 0" warning | Old `qe_score.py`. `git pull` for the new one |
| "GPU available: False" in the QE run | Normal on CPU. See Step 11 to enable the GPU (optional) |
| `502` from `--list-models` | The server or tunnel is down. Tell Bilal |
| `scp` says `Connection closed` | Retry. Check `ssh` works. Try `scp -O`. Check you are on the IITB network/VPN |
| `verify_data.py` says `MISMATCH` | Copy the two files from the server again (Step 8). Do not edit them or save them from Excel |
| `make_pairs.py` says files already exist | Good, it protects you. For your test set use a new `--out-dir`. Never use `--force` on the gold set |
| `'NoneType' object has no attribute 'shape'` (IndicTrans2) | Make sure `use_cache=False` is set in `load_indictrans2` in `mt_models.py` (it already is in this repo) |
| `torch.load` vulnerability / needs torch >= 2.6 | `pip install --upgrade torch` (GPU users: redo the CUDA install command) |
| `CUDA out of memory` | Test models one at a time instead of all together, or set `DEVICE = "cpu"` in `mt_models.py` |
| Qwen is very slow | Normal on CPU. Use `Qwen/Qwen2.5-0.5B-Instruct` in `download_models.py` for a lighter model |
| `UnicodeEncodeError` or boxes instead of Hindi | Read the output files in VS Code, not the terminal |
| Download stops halfway | Re-run the command, it resumes |
| Warnings about `torch_dtype`, truncation, generation flags, Lightning tips | Harmless, ignore them |

## Working together on GitHub

- Before you start working: **Fetch origin / Pull origin** in GitHub Desktop.
- When you finish something: commit with a clear message, then **Push origin**.
- Tell each other which file you're editing to avoid conflicts.
- **Before every push, look at the list of changed files.** You must not see anything from `data\`, `data_test\`,
  `outputs\`, `models\`, `venv\`, `venv_qe\`, any `.jsonl`, `.ckpt`, `.env`, or anything with a password, token or key.
  If you do, stop and fix `.gitignore` first.
- Nothing in GitHub should ever contain: Bilal's server password, the vLLM API key, a Hugging Face token (`hf_...`).
  GitHub rejects files over 100 MB, but small secrets slip through, so check by eye.
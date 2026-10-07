# CS 769 Project: Multi-Model English→Hindi Translation Combiner

Given an English sentence, we get Hindi translations from 5 different machine
translation (MT) models and later combine them into one best translation
(majority voting, quality estimation, and "Don't Rank, Combine!" fusion).

**This README covers phase 1 only: getting all 5 models running on your own
Windows computer.** Comparing, scoring and merging come later.

## The 5 models

| Key | Model | Hugging Face ID | Login needed? |
|---|---|---|---|
| `opus` | opus-mt-en-hi | `Helsinki-NLP/opus-mt-en-hi` | No |
| `nllb` | NLLB-200 (600M) | `facebook/nllb-200-distilled-600M` | No |
| `mbart50` | mBART-50 | `facebook/mbart-large-50-one-to-many-mmt` | No |
| `indictrans2` | IndicTrans2 (200M) | `ai4bharat/indictrans2-en-indic-dist-200M` | **Yes (gated)** |
| `qwen` | Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | No |

## What you need before starting

- Windows, VS Code, and **Command Prompt** as the terminal (all commands below are for cmd, not PowerShell)
- Python 3.10 or newer (tested on 3.13). Check with `python --version`
- About **15 GB free disk space** for the model downloads
- A good internet connection (the downloads are large)
- A free Hugging Face account (needed for IndicTrans2, see Step 5)
- An NVIDIA GPU is optional. Without one everything still works, just slower
- Git or GitHub Desktop, to get this repository

---

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

## Step 5: Hugging Face login (required for IndicTrans2)

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
others.

To test your own sentences, edit the `SENTENCES` list at the top of `test_models.py`.

You're done with phase 1 when all five models give sensible Hindi sentences.

---

## Project structure

```
CS769 Project\
  README.md
  requirements.txt
  .gitignore
  download_models.py     downloads the 5 models into models\
  mt_models.py           loads each model and exposes a translate function
  test_models.py         runs the models on sample sentences
  models\                (not in GitHub) downloaded model files
  outputs\               (not in GitHub) generated results
  venv\                  (not in GitHub) your virtual environment
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `'source' is not recognized` | Wrong OS command. Use `venv\Scripts\activate` |
| `'python' is not recognized` | Try `py` instead, or reinstall Python with "Add to PATH" ticked |
| `ModuleNotFoundError` | The venv is not active. Run `venv\Scripts\activate` and check for `(venv)` |
| `torch.cuda.is_available()` is `False` but you have an NVIDIA GPU | CPU-only torch was installed. Redo Step 3 (GPU install) |
| `Microsoft Visual C++ 14.0 or greater is required` | Install C++ Build Tools (Step 4), then open a NEW terminal |
| `cannot import name 'PreTrainedTokenizerBase'` | `pip install "transformers<5"` |
| `401 ... Access to model ... is restricted` | Complete Step 5 (access request, token, login) |
| `'NoneType' object has no attribute 'shape'` (IndicTrans2) | Make sure `use_cache=False` is set in `load_indictrans2` in `mt_models.py` (it already is in this repo) |
| `torch.load` vulnerability / needs torch >= 2.6 | `pip install --upgrade torch` (GPU users: redo the CUDA install command) |
| `CUDA out of memory` | Test models one at a time instead of all together, or set `DEVICE = "cpu"` in `mt_models.py` |
| Qwen is very slow | Normal on CPU. Use `Qwen/Qwen2.5-0.5B-Instruct` in `download_models.py` for a lighter model |
| `UnicodeEncodeError` or boxes instead of Hindi | Read `outputs\test_results.txt` in VS Code, not the terminal |
| Download stops halfway | Re-run `python download_models.py`, it resumes |
| Warnings about `torch_dtype`, truncation, or generation flags | Harmless, ignore them |

## Working together on GitHub

- Before you start working: **Fetch origin / Pull origin** in GitHub Desktop.
- When you finish something: commit with a clear message, then **Push origin**.
- Tell each other which file you're editing to avoid conflicts.
- Never commit `models\`, `venv\`, or your Hugging Face token. The `.gitignore` handles the first two. GitHub rejects files over 100 MB.
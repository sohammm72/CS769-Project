"""
Tiny test: checks that your API key + server address work, then translates
one sentence.

The address and key are read from environment variables (NOT written in this
file), so you can share/upload this script safely.

Usage:
    python test_api.py                 # uses the first model the server lists
    python test_api.py <model-id>      # test a specific model
"""

import os
import sys

import requests

sys.stdout.reconfigure(encoding="utf-8")  # so Hindi text prints correctly on Windows

BASE_URL = os.environ.get("VLLM_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("VLLM_API_KEY", "")

if not BASE_URL or not API_KEY:
    print("Missing VLLM_BASE_URL or VLLM_API_KEY. Set them first (see steps).")
    sys.exit(1)

headers = {"Authorization": f"Bearer {API_KEY}"}

# Step 1: ask the server which models it has
resp = requests.get(f"{BASE_URL}/v1/models", headers=headers, timeout=30)
print("Model list status:", resp.status_code)
resp.raise_for_status()
models = [m["id"] for m in resp.json()["data"]]
print("Models available:")
for m in models:
    print("  -", m)

# Step 2: translate one sentence
model = sys.argv[1] if len(sys.argv) > 1 else models[0]
print(f"\nTesting translation with: {model}")

payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": "Translate the following English text into Hindi. "
            "Reply with only the translation.\n\nThe weather is beautiful today.",
        }
    ],
    "temperature": 0,
    "max_tokens": 200,
}
resp = requests.post(
    f"{BASE_URL}/v1/chat/completions", headers=headers, json=payload, timeout=120
)
print("Translation status:", resp.status_code)
resp.raise_for_status()
print("Output:", resp.json()["choices"][0]["message"]["content"])
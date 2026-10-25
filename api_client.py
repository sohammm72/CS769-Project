"""
api_client.py - small client for the vLLM server Bilal hosts.

The server speaks the OpenAI-style API (/v1/models, /v1/chat/completions).
The address and key are NOT stored in this file. They are read from two
environment variables, so the file is safe to push to GitHub:

    Windows (Command Prompt):   set VLLM_BASE_URL=https://...
                                set VLLM_API_KEY=...
    Windows (PowerShell):       $env:VLLM_BASE_URL="https://..."
                                $env:VLLM_API_KEY="..."
    Mac / Linux:                export VLLM_BASE_URL="https://..."
                                export VLLM_API_KEY="..."

Needs only:  pip install requests
"""

import os
import re
import time

import requests


class ConfigError(Exception):
    """Address or key is missing."""


class ServerUnavailable(Exception):
    """The server is off, unreachable, or the model is not served."""


class ServerRejected(Exception):
    """The server answered, but refused the request (bad key, bad model name...)."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def normalise_base_url(url):
    """Accept the address however it was pasted and return a clean base URL.

    Strips spaces, quotes, a trailing slash, and a trailing /v1 or /v1/models,
    because the client adds those itself.
    """
    url = url.strip().strip('"').strip("'").strip()
    url = url.rstrip("/")
    url = re.sub(r"/v1(/models)?$", "", url)
    return url.rstrip("/")


class VLLMClient:
    def __init__(self, base_url, api_key, timeout=120, max_retries=3):
        if not base_url or not api_key:
            raise ConfigError(
                "Missing VLLM_BASE_URL or VLLM_API_KEY. Set both in this "
                "terminal window first (see the top of api_client.py)."
            )
        self.base_url = normalise_base_url(base_url)
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        # The key is only ever put in this header; it is never printed.
        self.session.headers.update(
            {
                "Authorization": "Bearer " + api_key.strip().strip('"').strip("'"),
                "Content-Type": "application/json",
            }
        )

    @classmethod
    def from_env(cls, **kwargs):
        return cls(
            os.environ.get("VLLM_BASE_URL", ""),
            os.environ.get("VLLM_API_KEY", ""),
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Server checks
    # ------------------------------------------------------------------ #
    def list_models(self):
        """Return the list of model ids the server currently serves."""
        resp = self._request("GET", "/v1/models", retries=1, timeout=30)
        return [m["id"] for m in resp.json().get("data", [])]

    def check_server(self):
        """Quick health check. Returns (ok, message). Never raises."""
        try:
            models = self.list_models()
        except ServerRejected as e:
            return False, str(e)
        except ServerUnavailable as e:
            return False, str(e)
        except Exception as e:  # unexpected, but do not crash a health check
            return False, "Unexpected error while checking the server: %s" % e
        if not models:
            return False, "The server is up but is not serving any model."
        return True, "Server is up. Models: " + ", ".join(models)

    # ------------------------------------------------------------------ #
    # Translation
    # ------------------------------------------------------------------ #
    def translate(
        self,
        model,
        text,
        source_lang="English",
        target_lang="Hindi",
        max_tokens=512,
        temperature=0.0,
    ):
        """Translate one sentence with one model. Returns the translation string."""
        prompt = (
            "Translate the following %s text into %s. "
            "Reply with the translation only, no explanations.\n\n%s"
            % (source_lang, target_lang, text)
        )
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = self._request("POST", "/v1/chat/completions", json=payload)
        try:
            out = resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError):
            raise ServerRejected("Unexpected response format from the server.")
        return (out or "").strip()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _request(self, method, path, retries=None, timeout=None, **kwargs):
        retries = self.max_retries if retries is None else retries
        timeout = self.timeout if timeout is None else timeout
        url = self.base_url + path
        last_problem = ""

        for attempt in range(1, retries + 1):
            try:
                resp = self.session.request(method, url, timeout=timeout, **kwargs)
            except (requests.ConnectionError, requests.Timeout) as e:
                last_problem = "could not reach the server (%s)" % type(e).__name__
            else:
                code = resp.status_code
                if code < 400:
                    return resp
                if code in (401, 403):
                    raise ServerRejected(
                        "The server rejected the key (HTTP %d). Check VLLM_API_KEY."
                        % code,
                        code,
                    )
                if code in (400, 404, 422):
                    raise ServerRejected(
                        "The server refused the request (HTTP %d): %s"
                        % (code, resp.text[:300]),
                        code,
                    )
                # 5xx and anything else: the server (or the tunnel in front of
                # it) is down or busy, which may be temporary.
                last_problem = "server error HTTP %d" % code

            if attempt < retries:
                time.sleep(min(2 ** attempt, 20))

        raise ServerUnavailable(
            "Server not available (%s). If this is a 502, the models are "
            "probably not launched - ask Bilal to start them." % last_problem
        )
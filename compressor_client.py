# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import requests

SERVER_URL = "http://localhost:1234/compress"
# TEXT = """
# The quick brown fox jumps over the lazy dog.
# Here is a very long prompt you want to compress before feeding it to the LLM …
# """
# RATE = 0.6                                      # compress rate（0‒1]

# payload = {
#     "text": TEXT,
#     "rate": RATE
# }


# curl -X POST "http://localhost:1234/compress"  -H "Content-Type: application/json" -d '{"text": "<|fim_prefix|>", "rate": 0.6}'


def compress_prompt(text, rate):
    payload = {
        "text": text,
        "rate": rate
    }
    try:
        resp = requests.post(SERVER_URL, json=payload, timeout=30)
        resp.raise_for_status()                     # HTTP 4xx/5xx
        data = resp.json()
        return data
    except requests.exceptions.RequestException as e:
        print("Network or server error:", e)
    except ValueError:
        print("Response is not valid JSON:", resp.text)
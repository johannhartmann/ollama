#!/usr/bin/env python3
"""Smoke test proving /api/multivectors output is consumable for late interaction.

It encodes a query and two documents, then scores each document against the
query with MaxSim:

    score(q, d) = sum_i max_j dot(q_i, d_j)

This is a developer validation tool, not a unit test: it requires a running
Ollama server and a real multivector (ColBERT/ModernColBERT) model. It does not
assert exact scores; it only checks that both query and documents return
matrices of matching dimension and that the scores are finite.

Usage:
    MODEL=multivector-validate-test scripts/multivector-maxsim-smoke.py

Optional:
    OLLAMA_HOST   ollama server (default http://127.0.0.1:11434)
"""

import json
import math
import os
import sys
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.environ.get("MODEL")

QUERY = "Welche Stadt ist die Hauptstadt von Deutschland?"
DOCS = [
    "Berlin ist die Hauptstadt von Deutschland.",
    "Paris ist die Hauptstadt von Frankreich.",
]


def multivector(text):
    body = json.dumps({"model": MODEL, "input": text}).encode()
    req = urllib.request.Request(
        f"{HOST}/api/multivectors",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        payload = json.load(resp)
    return payload["data"][0]["vectors"]


def maxsim(query, doc):
    total = 0.0
    for q in query:
        total += max(sum(a * b for a, b in zip(q, d)) for d in doc)
    return total


def main():
    if not MODEL:
        print("error: set MODEL to a multivector model name", file=sys.stderr)
        return 1

    q = multivector(QUERY)
    docs = [multivector(d) for d in DOCS]

    qdim = len(q[0]) if q else 0
    if qdim == 0:
        print("FAIL: query returned no rows", file=sys.stderr)
        return 1

    scores = []
    for text, d in zip(DOCS, docs):
        if not d:
            print(f"FAIL: document returned no rows: {text!r}", file=sys.stderr)
            return 1
        if len(d[0]) != qdim:
            print(
                f"FAIL: dimension mismatch query={qdim} doc={len(d[0])}",
                file=sys.stderr,
            )
            return 1
        score = maxsim(q, d)
        if not math.isfinite(score):
            print(f"FAIL: non-finite score for {text!r}", file=sys.stderr)
            return 1
        scores.append((score, text))

    ranked = sorted(scores, key=lambda s: s[0], reverse=True)
    print(f"query: {QUERY!r} (rows={len(q)}, dim={qdim})")
    for rank, (score, text) in enumerate(ranked, 1):
        print(f"  {rank}. score={score:.4f}  {text!r}")
    print("PASS: matrices are consumable for MaxSim late interaction")
    return 0


if __name__ == "__main__":
    sys.exit(main())

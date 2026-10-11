"""Deterministic, context-disjoint associative retrieval with an independent oracle."""
import hashlib
import json
import random

KEYS = 16
VALUES = 16
PAIRS = 4
LENGTH = 10
VOCAB = 33
SPLIT_SEEDS = {"train": 104729, "dev": 130363, "eval": 155921}


def context_bucket(tokens):
    # Canonical dictionary: reordered copies and different queries cannot cross splits.
    pairs = sorted(zip(tokens[:8:2], tokens[1:8:2]))
    raw = json.dumps(pairs, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % 100


def belongs(tokens, split):
    bucket = context_bucket(tokens)
    return {"train": bucket < 80, "dev": 80 <= bucket < 90,
            "eval": bucket >= 90}[split]


def oracle(tokens):
    if len(tokens) != LENGTH or any(type(t) is not int for t in tokens):
        raise ValueError("expected ten integer tokens")
    keys, values = tokens[:8:2], tokens[1:8:2]
    if (len(set(keys)) != PAIRS or any(not 0 <= k < KEYS for k in keys)
            or len(set(values)) != PAIRS or any(not 16 <= v < 32 for v in values)
            or tokens[8] != 32 or tokens[9] not in keys):
        raise ValueError("invalid retrieval example")
    dictionary = dict(zip(keys, values))
    answer = dictionary[tokens[9]] - KEYS
    position = next(i for i in range(1, 8, 2) if tokens[i - 1] == tokens[9])
    return answer, position


def generate(split, n):
    if split not in SPLIT_SEEDS or type(n) is not int or not 1 <= n <= 32768:
        raise ValueError("invalid split or dataset size")
    rng = random.Random(SPLIT_SEEDS[split])
    rows, seen = [], set()
    while len(rows) < n:
        keys = rng.sample(range(KEYS), PAIRS)
        values = rng.sample(range(16, 32), PAIRS)
        slot = len(rows) % PAIRS  # exactly balanced retrieval positions
        tokens = [t for pair in zip(keys, values) for t in pair] + [32, keys[slot]]
        canonical = tuple(sorted(zip(keys, values)))
        if canonical in seen or not belongs(tokens, split):
            continue
        seen.add(canonical)
        answer, position = oracle(tokens)
        rows.append({"id": f"{split}-{len(rows):05d}", "tokens": tokens,
                     "answer": answer, "value_position": position})
    return rows


def token_labels(tokens):
    return [f"K{t:02d}" if t < 16 else f"V{t-16:02d}" if t < 32 else "QUERY"
            for t in tokens]

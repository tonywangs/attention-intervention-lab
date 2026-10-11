"""Validate a frozen, bounded protocol before allocating models or datasets."""
from .model import Config


def validate_protocol(p):
    if not isinstance(p, dict) or p.get("schema") != 1 or p.get("frozen") is not True:
        raise ValueError("expected frozen protocol schema 1")
    Config(**p["architecture"]).validate()
    if p["training_seeds"] != [11, 29, 47]:
        raise ValueError("unexpected final seeds")
    if p["data"] != {"train_n": 16384, "dev_n": 512, "eval_n": 512,
                     "generator": "canonical-dictionary-sha256-v1", "seeds": {"train": 104729, "dev": 130363, "eval": 155921}}:
        raise ValueError("unexpected data specification")
    t = p["training"]
    if (set(t) != {"steps", "batch", "lr", "validate_every"}
            or type(t["steps"]) is not int or not 1 <= t["steps"] <= 10000
            or t["batch"] != 128 or t["lr"] != 0.001 or t["validate_every"] != 100):
        raise ValueError("invalid training budget")
    if p["budgets"] != {"pilot_seconds": 1800, "training_seconds": 10800,
                         "evaluation_seconds": 600, "attention_examples": 16,
                         "bootstrap_replicates": 2000, "bootstrap_seed": 20261011}:
        raise ValueError("unexpected resource budgets")
    if p["tolerances"] != {"logit_atol": 0.000002, "logit_rtol": 0.000002,
                            "probability_atol": 0.000002, "noop_atol": 0.0}:
        raise ValueError("unexpected replay tolerances")

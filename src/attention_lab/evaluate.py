"""Offline checkpoint evaluation with explicit budgets and bounded artifact schemas."""
from pathlib import Path
import hashlib
import resource
import time
import numpy as np
import torch
from .artifacts import (Budget, canonical, manifest, read_json, sha256,
                        transaction, verify_manifest, write_json)
from .data import generate, oracle, belongs
from .model import Config, Transformer, load_weights
from .protocol import validate_protocol
from .statistics import from_logits, metrics, bootstrap_indices, paired_changes, rank_comparison
from .train import configure, environment, score, tensors


def validate_rows(rows, split, count):
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("wrong dataset size")
    seen = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"id", "tokens", "answer", "value_position"}:
            raise ValueError("invalid dataset record")
        if row["id"] != f"{split}-{i:05d}" or oracle(row["tokens"]) != (row["answer"], row["value_position"]):
            raise ValueError("dataset answer or identity mismatch")
        if not belongs(row["tokens"], split):
            raise ValueError("dataset split leakage")
        key = tuple(sorted(zip(row["tokens"][:8:2], row["tokens"][1:8:2])))
        if key in seen:
            raise ValueError("duplicate dictionary")
        seen.add(key)
    return rows


def state_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def infer(model, x, budget, intervention=None, noop=False):
    logits, patterns = [], []
    for batch in x.split(128):
        budget.check()
        l, p = model(batch, intervention, noop)
        logits.append(l.numpy())
        patterns.append(p.numpy())
    return np.concatenate(logits), np.concatenate(patterns)


def baselines(rows):
    first = np.mean([r["tokens"][1]-16 == r["answer"] for r in rows])
    return {"uniform_vocabulary": {"expected_accuracy": 1/16, "loss": float(np.log(16)), "p_correct": 1/16},
            "uniform_context_values": {"expected_accuracy": 1/4, "loss": float(np.log(4)), "p_correct": 1/4},
            "first_context_value": {"accuracy": float(first), "loss": None,
                                    "loss_note": "Infinite cross-entropy for a deterministic wrong prediction; encoded as null."},
            "dictionary_oracle": {"accuracy": 1.0, "loss": 0.0, "p_correct": 1.0}}


def evaluate(training_dir, destination, max_seconds=600, evaluation_rows=None):
    if not 0 < max_seconds <= 600:
        raise ValueError("evaluation budget must be in (0, 600] seconds")
    budget = Budget(max_seconds)
    configure()
    training_dir = Path(training_dir)
    training_manifest = verify_manifest(training_dir)
    p = read_json(training_dir / "protocol.json", 65536)
    validate_protocol(p)
    training = read_json(training_dir / "training.json", 1024*1024)
    if training["protocol_sha256"] != sha256(training_dir / "protocol.json"):
        raise ValueError("training protocol hash mismatch")
    if [r["seed"] for r in training["records"]] != p["training_seeds"]:
        raise ValueError("training seed mismatch")
    dev = validate_rows(read_json(training_dir / "data/dev.json"), "dev", 512)
    rows = generate("eval", 512) if evaluation_rows is None else evaluation_rows
    validate_rows(rows, "eval", 512)
    x, y = tensors(rows)
    dx, dy = tensors(dev)
    answers = y.numpy()
    cfg = Config(**p["architecture"])
    names = [f"L{layer}H{head}" for layer in range(cfg.layers) for head in range(cfg.heads)]
    indices = bootstrap_indices(len(rows), p["budgets"]["bootstrap_replicates"], p["budgets"]["bootstrap_seed"])
    summary = {"schema": 1, "protocol_sha256": sha256(training_dir / "protocol.json"),
               "n": len(rows), "heads": names, "baselines": baselines(rows), "seeds": [],
               "attention_examples": 16,
               "uncertainty": "95% paired percentile bootstrap, 2000 resamples; within seed, unadjusted, conditional on model and constructed evaluation distribution.",
               "limitations": "Synthetic fixed-length task; three seeds; zero ablation may be out of distribution; attention and ablation are not complete mechanistic explanations."}
    with transaction(destination) as out:
        write_json(out / "eval.json", rows)
        summary["eval_sha256"] = sha256(out / "eval.json")
        for record in training["records"]:
            budget.check()
            seed = record["seed"]
            checkpoint = training_dir / f"seed-{seed}.npz"
            if sha256(checkpoint) != record["checkpoint_sha256"]:
                raise ValueError("selected checkpoint hash mismatch")
            model = Transformer(cfg).eval()
            load_weights(model, checkpoint)
            reproduced_dev = score(model, dx, dy)
            if (reproduced_dev["accuracy"] != record["selected_dev"]["accuracy"] or
                    abs(reproduced_dev["loss"]-record["selected_dev"]["loss"]) > 2e-6):
                raise ValueError("checkpoint fails development reproduction")
            before = state_hash(model)
            intact_logits, pattern = infer(model, x, budget)
            noop_logits, _ = infer(model, x, budget, noop=True)
            if not np.array_equal(intact_logits, noop_logits):
                raise ValueError("no-op logits differ")
            all_logits = {"intact": intact_logits, "noop": noop_logits}
            for layer in range(cfg.layers):
                for head in range(cfg.heads):
                    logits, _ = infer(model, x, budget, (layer, head))
                    all_logits[f"L{layer}H{head}"] = logits
            if state_hash(model) != before:
                raise ValueError("interventions changed model weights")
            values = {name: from_logits(logits, answers) for name, logits in all_logits.items()}
            # Preserve each example's score for independent reaggregation.
            mass = np.stack([pattern[i, :, :, -1, r["value_position"]].reshape(-1)
                             for i, r in enumerate(rows)])
            conditions = []
            for name, v in values.items():
                budget.check()
                condition = {"name": name, **metrics(v)}
                condition["paired"] = paired_changes(values["intact"], v, indices)
                condition["by_value_position"] = {
                    str(pos): metrics({k: a[[r["value_position"] == pos for r in rows]] for k, a in v.items()})
                    for pos in (1, 3, 5, 7)}
                conditions.append(condition)
            effects = [c["paired"]["probability_drop"]["mean"] for c in conditions[2:]]
            seed_summary = {"seed": seed, "selected_step": record["selected_step"],
                            "dev_accuracy": reproduced_dev["accuracy"],
                            "learning_gate": reproduced_dev["accuracy"] >= 0.9,
                            "interpretation": "task-mechanism gate passed" if reproduced_dev["accuracy"] >= 0.9 else "learning-floor diagnosis only",
                            "checkpoint_sha256": sha256(checkpoint), "weights_unchanged": True,
                            "noop_max_logit_difference": 0.0, "conditions": conditions,
                            "ranking": rank_comparison(mass.astype(np.float64).mean(axis=0), effects, names)}
            example_results = []
            for i, row in enumerate(rows):
                example_results.append({**row, "attention_mass": mass[i].tolist(),
                    "conditions": {name: {"logits": logits[i].tolist(),
                        "prediction": int(values[name]["prediction"][i]),
                        "correct": bool(values[name]["correct"][i]),
                        "loss": float(values[name]["loss"][i]),
                        "p_correct": float(values[name]["p_correct"][i])}
                        for name, logits in all_logits.items()}})
            write_json(out / f"seed-{seed}.json", {"seed": seed, "examples": example_results})
            write_json(out / f"attention-{seed}.json", {"seed": seed, "examples": [
                {"id": rows[i]["id"], "pattern": pattern[i].tolist()} for i in range(16)]})
            summary["seeds"].append(seed_summary)
        valid = [s["ranking"]["spearman"] for s in summary["seeds"] if s["ranking"]["spearman"] is not None]
        summary["spearman_range"] = [min(valid), max(valid)] if valid else None
        summary["all_learning_gates_passed"] = all(s["learning_gate"] for s in summary["seeds"])
        write_json(out / "summary.json", summary)
        from .report import render_directory
        render_directory(out, out / "report.html")
        budget.check()
        write_json(out / "provenance.json", {"environment": environment(),
                   "elapsed_seconds": time.monotonic() - budget.start,
                   "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "training_manifest_sha256": sha256(training_dir / "manifest.json"),
                   "protocol_sha256": summary["protocol_sha256"],
                   "evaluation_budget_seconds": max_seconds})
        write_json(out / "manifest.json", manifest(out))
    return summary


def replay(training_dir, reference_dir, destination, max_seconds=600):
    reference_dir = Path(reference_dir)
    verify_manifest(reference_dir)
    rows = read_json(reference_dir / "eval.json")
    # Stage both reevaluation and comparison: mismatch must not leave a successful output.
    with transaction(destination) as out:
        evaluate(training_dir, out / "evaluation", max_seconds, evaluation_rows=rows)
        generated = out / "evaluation"
        p = read_json(Path(training_dir) / "protocol.json")
        checks = []
        for seed in p["training_seeds"]:
            old = read_json(reference_dir / f"seed-{seed}.json")
            new = read_json(generated / f"seed-{seed}.json")
            if len(old["examples"]) != 512 or len(new["examples"]) != 512:
                raise ValueError("replay example count mismatch")
            for a, b in zip(old["examples"], new["examples"]):
                if any(a[k] != b[k] for k in ("id", "tokens", "answer", "value_position")):
                    raise ValueError("replay example identity mismatch")
                if set(a["conditions"]) != set(b["conditions"]):
                    raise ValueError("replay condition mismatch")
                for name in a["conditions"]:
                    left, right = a["conditions"][name], b["conditions"][name]
                    if (left["prediction"] != right["prediction"] or left["correct"] != right["correct"]
                            or not np.allclose(left["logits"], right["logits"], atol=2e-6, rtol=2e-6)
                            or abs(left["p_correct"]-right["p_correct"]) > 2e-6):
                        raise ValueError("checkpoint replay differs beyond frozen tolerances")
            files = [f"seed-{seed}.json", f"attention-{seed}.json"]
            checks.extend({"file": f, "byte_identical": sha256(reference_dir/f) == sha256(generated/f)} for f in files)
        checks.extend({"file": f, "byte_identical": sha256(reference_dir/f) == sha256(generated/f)}
                      for f in ("summary.json", "eval.json", "report.html"))
        # Compare all saved numeric evidence, not only argmax predictions. A modified
        # loss, aggregate, rank, or attention matrix must not silently pass replay.
        def compare(left, right):
            if type(left) is not type(right):
                raise ValueError("replay artifact schema differs")
            if isinstance(left, dict):
                if set(left) != set(right):
                    raise ValueError("replay artifact keys differ")
                for key in left:
                    compare(left[key], right[key])
            elif isinstance(left, list):
                if len(left) != len(right):
                    raise ValueError("replay artifact shape differs")
                for a, b in zip(left, right):
                    compare(a, b)
            elif type(left) is float:
                if not np.isclose(left, right, atol=2e-6, rtol=2e-6):
                    raise ValueError("replay numeric evidence differs")
            elif left != right:
                raise ValueError("replay artifact value differs")
        for check in checks:
            if check["file"].endswith(".json"):
                compare(read_json(reference_dir / check["file"]),
                        read_json(generated / check["file"]))
        write_json(out / "replay.json", {"predictions_match": True, "within_frozen_tolerances": True,
                   "all_deterministic_files_identical": all(c["byte_identical"] for c in checks), "checks": checks})
        write_json(out / "manifest.json", manifest(out))

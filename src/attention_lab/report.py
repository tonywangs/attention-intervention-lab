"""Deterministic, self-contained report. External data is only ever inert JSON/text."""
import base64
import hashlib
from importlib.resources import files
import math
from pathlib import Path
from .artifacts import canonical, read_json, MAX_FILE
from .data import token_labels


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_payload(payload):
    summary, examples, attention = payload["summary"], payload["examples"], payload["attention"]
    if (summary["schema"] != 1 or summary["n"] != 512
            or summary["heads"] != [f"L{l}H{h}" for l in range(2) for h in range(4)]
            or len(summary["seeds"]) != 3 or summary["attention_examples"] != 16
            or len(str(summary["limitations"])) > 2048):
        raise ValueError("unsupported report bounds")
    if [s["seed"] for s in summary["seeds"]] != [11, 29, 47]:
        raise ValueError("unsupported report seeds")
    for seed in ("11", "29", "47"):
        rows, matrices = examples[seed], attention[seed]
        if len(rows) != 16 or len(matrices) != 16:
            raise ValueError("report example bounds exceeded")
        for row, matrix in zip(rows, matrices):
            if (len(row["labels"]) != 10 or any(not isinstance(t, str) or len(t) > 128 for t in row["labels"])
                    or not isinstance(row["id"], str) or len(row["id"]) > 128
                    or row["id"] != matrix["id"] or row["answer"] not in range(16)
                    or row["value_position"] not in (1, 3, 5, 7)
                    or set(row["conditions"]) != {"intact", "noop", *summary["heads"]}):
                raise ValueError("invalid report example")
            for c in row["conditions"].values():
                if (len(c["logits"]) != 16 or any(not _number(v) for v in c["logits"])
                        or c["prediction"] not in range(16) or not _number(c["p_correct"])
                        or not 0 <= c["p_correct"] <= 1 or not _number(c["loss"])):
                    raise ValueError("invalid report prediction")
            pattern = matrix["pattern"]
            if len(pattern) != 2:
                raise ValueError("report layer bounds")
            for layer in pattern:
                if len(layer) != 4:
                    raise ValueError("report head bounds")
                for head in layer:
                    if len(head) != 10 or any(len(r) != 10 for r in head):
                        raise ValueError("report matrix bounds")
                    for i, r in enumerate(head):
                        if (any(not _number(v) or not 0 <= v <= 1 for v in r)
                                or abs(sum(r)-1) > 1e-5 or any(v != 0 for v in r[i+1:])):
                            raise ValueError("invalid attention probability or causal mask")
    for seed in summary["seeds"]:
        if [c["name"] for c in seed["conditions"]] != ["intact", "noop", *summary["heads"]]:
            raise ValueError("invalid summary conditions")
        for c in seed["conditions"]:
            if any(not _number(c[k]) for k in ("accuracy", "loss", "p_correct")):
                raise ValueError("invalid summary metric")
            for metric in ("accuracy_drop", "loss_increase", "probability_drop"):
                pair = c["paired"][metric]
                if len(pair["ci95"]) != 2 or any(not _number(v) for v in [pair["mean"], *pair["ci95"]]):
                    raise ValueError("invalid uncertainty")
        rank = seed["ranking"]
        for field in ("attention_mean", "probability_drop_mean", "attention_rank", "effect_rank"):
            if len(rank[field]) != 8 or any(not _number(v) for v in rank[field]):
                raise ValueError("invalid ranking")
        if rank["spearman"] is not None and (not _number(rank["spearman"]) or not -1 <= rank["spearman"] <= 1):
            raise ValueError("invalid correlation")


def make_payload(directory):
    directory = Path(directory)
    payload = {"summary": read_json(directory / "summary.json"), "examples": {}, "attention": {}}
    for seed in (11, 29, 47):
        rows = read_json(directory / f"seed-{seed}.json")["examples"]
        if len(rows) != 512:
            raise ValueError("wrong prediction count")
        payload["examples"][str(seed)] = [{**r, "labels": token_labels(r["tokens"])} for r in rows[:16]]
        payload["attention"][str(seed)] = read_json(directory / f"attention-{seed}.json")["examples"]
    validate_payload(payload)
    return payload


def render(payload):
    validate_payload(payload)
    template = files("attention_lab").joinpath("report.html").read_text()
    script = template.split('<script id="app">', 1)[1].split('</script>', 1)[0]
    script_hash = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    raw = canonical(payload).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = template.replace("__SCRIPT_HASH__", script_hash).replace("__DATA__", raw)
    if len(html.encode()) > MAX_FILE:
        raise ValueError("report file budget exceeded")
    return html


def render_directory(directory, destination):
    Path(destination).write_text(render(make_payload(directory)))

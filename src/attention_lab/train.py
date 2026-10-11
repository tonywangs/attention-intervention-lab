"""Development-only model selection and bounded three-seed training."""
from dataclasses import asdict
import hashlib
import platform
import resource
import time
from pathlib import Path
import numpy as np
import torch
from .artifacts import Budget, canonical, manifest, read_json, sha256, transaction, write_json
from .data import generate
from .model import Config, Transformer, load_weights, save_weights


def configure(seed=0):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)


def tensors(rows):
    return (torch.tensor([r["tokens"] for r in rows], dtype=torch.long),
            torch.tensor([r["answer"] for r in rows], dtype=torch.long))


@torch.no_grad()
def score(model, x, y):
    model.eval()
    logits = torch.cat([model(batch)[0] for batch in x.split(128)])
    return {"accuracy": float((logits.argmax(-1) == y).float().mean()),
            "loss": float(torch.nn.functional.cross_entropy(logits, y))}


def environment():
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "platform": platform.platform(),
            "machine": platform.machine(), "threads": torch.get_num_threads(),
            "cpu": next((line.split(":", 1)[1].strip() for line in
                         Path("/proc/cpuinfo").read_text().splitlines()
                         if line.startswith("model name")), "unknown")}


def train_one(cfg, train, dev, seed, settings, directory, budget):
    configure(seed)
    started = time.monotonic()
    model = Transformer(cfg)
    x, y = tensors(train)
    dx, dy = tensors(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=0.01)
    rng = torch.Generator().manual_seed(seed + 1000000)
    history, best = [], (-1.0, float("-inf"))
    checkpoint = directory / f"seed-{seed}.npz"
    initial = score(model, dx, dy)
    timed_out = False
    for step in range(1, settings["steps"] + 1):
        try:
            budget.check()
        except TimeoutError:
            timed_out = True
            break
        model.train()
        idx = torch.randint(len(train), (settings["batch"],), generator=rng)
        logits, _ = model(x[idx])
        loss = torch.nn.functional.cross_entropy(logits, y[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % settings["validate_every"] == 0 or step == settings["steps"]:
            metrics = score(model, dx, dy)
            metrics.update({"step": step, "elapsed_seconds": time.monotonic() - started,
                            "batch_loss": float(loss.detach())})
            history.append(metrics)
            key = (metrics["accuracy"], -metrics["loss"])
            if key > best:
                best, best_step = key, step
                save_weights(model, checkpoint)
            print(f"seed={seed} step={step} dev={metrics['accuracy']:.4f} loss={metrics['loss']:.4f}", flush=True)
    if not checkpoint.exists():
        # Preserve initial/partial failed run without pretending it completed training.
        save_weights(model, checkpoint)
        metrics = score(model, dx, dy)
        best, best_step = (metrics["accuracy"], -metrics["loss"]), max(0, step - 1)
    load_weights(model, checkpoint)
    selected = score(model, dx, dy)
    record = {"seed": seed, "initial_dev": initial, "history": history,
              "selected_step": best_step, "selected_dev": selected,
              "learning_gate": selected["accuracy"] >= 0.90,
              "timed_out": timed_out, "parameters": sum(p.numel() for p in model.parameters()),
              "elapsed_seconds": time.monotonic() - started,
              "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "checkpoint_sha256": sha256(checkpoint)}
    write_json(directory / f"seed-{seed}.json", record)
    return record


def pilot(destination, steps=1500):
    if not 1 <= steps <= 3000:
        raise ValueError("pilot steps out of range")
    budget = Budget(1800)
    configure()
    with transaction(destination) as out:
        train, dev = generate("train", 16384), generate("dev", 512)
        settings = {"steps": steps, "batch": 128, "lr": 0.001, "validate_every": 100}
        cfg = Config()
        record = train_one(cfg, train, dev, 7, settings, out, budget)
        write_json(out / "pilot.json", {"architecture": asdict(cfg), "settings": settings,
                   "seed": 7, "eval_accessed": False, "environment": environment(),
                   "elapsed_seconds": time.monotonic() - budget.start, "result": record})
        write_json(out / "manifest.json", manifest(out))


def train_final(protocol_path, destination):
    from .protocol import validate_protocol
    protocol = read_json(protocol_path, 65536)
    validate_protocol(protocol)
    budget = Budget(protocol["budgets"]["training_seconds"])
    configure()
    with transaction(destination) as out:
        write_json(out / "protocol.json", protocol)
        train, dev = generate("train", protocol["data"]["train_n"]), generate("dev", protocol["data"]["dev_n"])
        # Dataset shards stay well below the per-file limit.
        data_dir = out / "data"
        data_dir.mkdir()
        for offset in range(0, len(train), 2048):
            write_json(data_dir / f"train-{offset//2048:02d}.json", train[offset:offset+2048])
        write_json(data_dir / "dev.json", dev)
        records = []
        for seed in protocol["training_seeds"]:
            records.append(train_one(Config(**protocol["architecture"]), train, dev, seed,
                                     protocol["training"], out, budget))
        write_json(out / "training.json", {"protocol_sha256": sha256(protocol_path),
                   "records": records, "environment": environment(),
                   "elapsed_seconds": time.monotonic() - budget.start,
                   "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "eval_accessed": False})
        write_json(out / "manifest.json", manifest(out))

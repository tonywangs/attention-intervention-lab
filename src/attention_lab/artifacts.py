"""Bounded JSON, hashes, and cancellation-safe output transactions."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time

MAX_FILE = 10 * 1024 * 1024
MAX_TREE = 32 * 1024 * 1024


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False) + "\n"


def write_json(path, obj):
    raw = canonical(obj).encode()
    if len(raw) > MAX_FILE:
        raise ValueError("JSON exceeds file budget")
    Path(path).write_bytes(raw)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unique(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError("duplicate JSON key")
        d[k] = v
    return d


def read_json(path, limit=MAX_FILE):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("invalid file or file budget exceeded")
    with path.open("rb") as f:
        raw = f.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("file budget exceeded")
    try:
        return json.loads(raw, object_pairs_hook=_unique,
                          parse_constant=lambda s: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (RecursionError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON") from exc


class Cancelled(Exception):
    pass


class Budget:
    def __init__(self, seconds):
        if not 0 < seconds <= 10800:
            raise ValueError("time budget must be in (0, 10800] seconds")
        self.start = time.monotonic()
        self.seconds = seconds

    def check(self):
        if time.monotonic() - self.start >= self.seconds:
            raise TimeoutError("execution time budget exhausted")


@contextmanager
def transaction(destination):
    """Reserve name exclusively; atomically replace the empty reservation on success."""
    destination = Path(destination).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()  # collisions (including symlinks) are errors, never overwrite
    staging = None
    handlers = {}
    try:
        def cancel(signum, frame):
            raise Cancelled(f"cancelled by signal {signum}")
        for signum in (signal.SIGTERM, signal.SIGINT):
            handlers[signum] = signal.signal(signum, cancel)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        yield staging
        files = list(staging.rglob("*"))
        sizes = [p.stat().st_size for p in files if p.is_file()]
        if len(sizes) > 1000 or sum(sizes) > MAX_TREE or any(s > MAX_FILE for s in sizes):
            raise ValueError("artifact publication budget exceeded")
        os.replace(staging, destination)
        staging = None
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        # Only the empty reservation can be removed. Successful output is nonempty.
        try:
            destination.rmdir()
        except OSError:
            pass


def manifest(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(directory.rglob("*")) if p.is_file() and p != directory / "manifest.json"}


def verify_manifest(directory):
    directory = Path(directory)
    saved = read_json(directory / "manifest.json", 256 * 1024)
    if not isinstance(saved, dict) or not 1 <= len(saved) <= 1000:
        raise ValueError("invalid manifest")
    total = 0
    for name, item in saved.items():
        p = Path(name)
        if p.is_absolute() or ".." in p.parts or "\\" in name or name == "manifest.json":
            raise ValueError("unsafe artifact path")
        path = directory / p
        if any(q.is_symlink() for q in [path, *path.parents]):
            raise ValueError("symlink artifact")
        if (not isinstance(item, dict) or set(item) != {"sha256", "bytes"}
                or type(item["bytes"]) is not int or not 0 <= item["bytes"] <= MAX_FILE
                or not path.is_file() or path.stat().st_size != item["bytes"]):
            raise ValueError("artifact size mismatch")
        total += item["bytes"]
        if total > MAX_TREE or sha256(path) != item["sha256"]:
            raise ValueError("artifact hash or total size mismatch")
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if actual != set(saved) | {"manifest.json"}:
        raise ValueError("unlisted artifact")
    return saved

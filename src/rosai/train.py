"""Train through an injected backend and write the run manifest."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from rosai.prepare import Prepared


@dataclass(frozen=True)
class Run:
    model_id: str
    thinking: bool
    seed: int
    loss: str
    config_hash: str
    max_seq_length: int = 2048
    limit: int | None = None
    backend: str = "fake"
    batch_size: int = 1


@dataclass(frozen=True)
class Manifest:
    model_id: str
    thinking: bool
    seed: int
    loss: str
    config_hash: str
    max_seq_length: int
    limit: int | None
    backend: str
    batch_size: int
    example_count: int


def load_run(path: str | Path) -> Run:
    raw_bytes = Path(path).read_bytes()
    raw = yaml.safe_load(raw_bytes)
    model = raw["model"]
    limit = raw.get("limit")
    return Run(
        model_id=str(model["id"]),
        thinking=bool(model["thinking"]),
        seed=int(raw["seed"]),
        loss=str(raw["loss"]),
        config_hash=hashlib.sha256(raw_bytes).hexdigest(),
        max_seq_length=int(model.get("max_seq_length", 2048)),
        limit=None if limit is None else int(limit),
        backend=str(raw.get("backend") or "fake"),
        batch_size=int(raw.get("batch_size", 1)),
    )


class FakeBackend:
    """Records the batch. It does not load a model."""

    def __init__(self) -> None:
        self.example_count = 0
        self.completion_only = False

    def fit(self, examples, *, completion_only: bool) -> None:
        self.example_count = len(tuple(examples))
        self.completion_only = completion_only


def train(prepared: Prepared, run: Run, backend, dest: str | Path) -> Manifest:
    destination = Path(dest)
    destination.mkdir(parents=True, exist_ok=True)
    examples = prepared.train
    if run.limit is not None:
        examples = examples[: run.limit]
    backend.fit(examples, completion_only=run.loss == "completion")
    manifest = Manifest(
        model_id=run.model_id,
        thinking=run.thinking,
        seed=run.seed,
        loss=run.loss,
        config_hash=run.config_hash,
        max_seq_length=run.max_seq_length,
        limit=run.limit,
        backend=run.backend,
        batch_size=run.batch_size,
        example_count=len(examples),
    )
    payload = {
        "model_id": manifest.model_id,
        "thinking": manifest.thinking,
        "seed": manifest.seed,
        "loss": manifest.loss,
        "config_hash": manifest.config_hash,
        "max_seq_length": manifest.max_seq_length,
        "limit": manifest.limit,
        "backend": manifest.backend,
        "batch_size": manifest.batch_size,
        "example_count": manifest.example_count,
    }
    (destination / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest

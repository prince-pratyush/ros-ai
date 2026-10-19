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


@dataclass(frozen=True)
class Manifest:
    model_id: str
    thinking: bool
    seed: int
    loss: str
    config_hash: str


def load_run(path: str | Path) -> Run:
    raw_bytes = Path(path).read_bytes()
    raw = yaml.safe_load(raw_bytes)
    model = raw["model"]
    return Run(
        model_id=str(model["id"]),
        thinking=bool(model["thinking"]),
        seed=int(raw["seed"]),
        loss=str(raw["loss"]),
        config_hash=hashlib.sha256(raw_bytes).hexdigest(),
    )


def train(prepared: Prepared, run: Run, backend, dest: str | Path) -> Manifest:
    destination = Path(dest)
    destination.mkdir(parents=True, exist_ok=True)
    examples = (*prepared.train, *prepared.dev, *prepared.held_out)
    backend.fit(examples, completion_only=run.loss == "completion")
    manifest = Manifest(
        model_id=run.model_id,
        thinking=run.thinking,
        seed=run.seed,
        loss=run.loss,
        config_hash=run.config_hash,
    )
    payload = {
        "model_id": manifest.model_id,
        "thinking": manifest.thinking,
        "seed": manifest.seed,
        "loss": manifest.loss,
        "config_hash": manifest.config_hash,
    }
    (destination / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest

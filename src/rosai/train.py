"""Train through an injected backend and write the run manifest."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from rosai.prepare import InstructionPair, Prepared


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


class TrainingError(Exception):
    """A run cannot be trained as configured."""


# Attention and MLP projections. Unsloth's Qwen QLoRA recipe.
_QLORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class QloraRequest:
    model_id: str
    max_seq_length: int
    seed: int
    batch_size: int
    epochs: int
    texts: tuple[str, ...]
    completion_only: bool
    load_in_4bit: bool
    target_modules: tuple[str, ...]
    lora_r: int
    lora_alpha: int
    adapter_dir: str
    output_dir: str
    resume_step: int
    checkpoint_every: int


def format_pair(pair: InstructionPair, *, thinking: bool) -> str:
    """Qwen3 chat text. Thinking stays off, so the assistant turn is an empty think block."""
    if thinking:
        raise TrainingError("Qwen3 QLoRA keeps thinking off")
    question = pair.messages[0][1]
    return (
        "<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
        f"{pair.completion}<|im_end|>\n"
    )


class UnslothBackend:
    """4-bit QLoRA through a session. The default session imports unsloth on fit."""

    def __init__(
        self,
        run: Run,
        dest: str | Path,
        *,
        epochs: int,
        checkpoint_every: int,
        start_step: int = 0,
        session=None,
        on_checkpoint=None,
    ) -> None:
        self.run = run
        self.dest = Path(dest)
        self.epochs = epochs
        self.checkpoint_every = checkpoint_every
        self.start_step = start_step
        self.session = session
        self.on_checkpoint = on_checkpoint
        self.step = start_step

    def fit(self, examples, *, completion_only: bool) -> None:
        if self.run.thinking:
            raise TrainingError("Qwen3 QLoRA keeps thinking off")
        texts = tuple(format_pair(example, thinking=False) for example in examples)
        adapter_dir = self.dest / "adapter"
        request = QloraRequest(
            model_id=self.run.model_id,
            max_seq_length=self.run.max_seq_length,
            seed=self.run.seed,
            batch_size=self.run.batch_size,
            epochs=self.epochs,
            texts=texts,
            completion_only=completion_only,
            load_in_4bit=True,
            target_modules=_QLORA_TARGETS,
            lora_r=16,
            lora_alpha=16,
            adapter_dir=str(adapter_dir),
            output_dir=str(self.dest / "trainer"),
            resume_step=self.start_step,
            checkpoint_every=self.checkpoint_every,
        )
        if self.session is None:
            self.step = fit_qlora(request, on_checkpoint=self.on_checkpoint)
        else:
            self.step = self.session.fit(request, on_checkpoint=self.on_checkpoint)
        saved = adapter_dir / "adapter_model.safetensors"
        config = adapter_dir / "adapter_config.json"
        if not saved.is_file() or not config.is_file():
            raise TrainingError("QLoRA session did not save the LoRA adapter")


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

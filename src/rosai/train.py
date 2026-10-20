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
        if not texts:
            raise TrainingError("no training pairs")
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


def resume_checkpoint(output_dir: str | Path, resume_step: int) -> str | bool:
    """Latest Hugging Face checkpoint, or False when this run should start fresh."""
    if resume_step <= 0:
        return False
    output = Path(output_dir)
    if not output.is_dir():
        return False
    found = []
    for path in output.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        suffix = path.name.split("-", 1)[1]
        if not suffix.isdigit():
            continue
        found.append((int(suffix), path))
    if not found:
        return False
    found.sort()
    return str(found[-1][1])


def on_save_checkpoint(directory: str | Path, step: int, commit) -> None:
    """Record a preemptible step, then commit the volume if a commit hook was given."""
    from rosai.modal_app import record_checkpoint

    record_checkpoint(directory, step)
    if commit is not None:
        commit()


def fit_qlora(request: QloraRequest, on_checkpoint=None) -> int:
    """Train a 4-bit QLoRA adapter. unsloth and torch load here, not at import."""
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=request.model_id,
        max_seq_length=request.max_seq_length,
        dtype=None,
        load_in_4bit=request.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=request.lora_r,
        target_modules=list(request.target_modules),
        lora_alpha=request.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=request.seed,
        use_rslora=False,
        loftq_config=None,
    )
    dataset = Dataset.from_dict({"text": list(request.texts)})
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=request.output_dir,
            max_length=request.max_seq_length,
            dataset_text_field="text",
            per_device_train_batch_size=request.batch_size,
            num_train_epochs=request.epochs,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=request.seed,
            save_strategy="steps",
            save_steps=request.checkpoint_every,
            report_to="none",
            packing=False,
        ),
    )
    if request.completion_only:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    class _Commit(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            on_save_checkpoint(
                Path(request.adapter_dir).parent,
                int(state.global_step),
                on_checkpoint,
            )

    trainer.add_callback(_Commit())
    stats = trainer.train(
        resume_from_checkpoint=resume_checkpoint(
            request.output_dir, request.resume_step
        )
    )
    Path(request.adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(request.adapter_dir)
    tokenizer.save_pretrained(request.adapter_dir)
    return int(stats.global_step)

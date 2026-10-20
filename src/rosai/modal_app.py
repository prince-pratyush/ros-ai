"""Preemptible Modal train plan. The GPU stays off until modal run."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from rosai.cost import guard


@dataclass(frozen=True)
class TrainPlan:
    gpu: str
    seconds: int
    cpu_cores: int
    memory_gib: int
    max_usd: float
    max_gpu_hours: float
    retries: int
    limit: int | None
    epochs: int
    estimate_usd: float
    start_step: int
    volume_name: str
    checkpoint_every: int


def plan_train(
    path: str | Path, checkpoint_dir: str | Path | None = None
) -> TrainPlan:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    seconds = int(raw["gpu_seconds"])
    cpu_cores = int(raw["cpu_cores"])
    memory_gib = int(raw["memory_gib"])
    gpu = str(raw["gpu"])
    max_usd = float(raw["max_usd"])
    max_gpu_hours = float(raw["max_gpu_hours"])
    retries = _retry_count(raw)
    usd = guard(
        gpu=gpu,
        seconds=seconds,
        cpu_cores=cpu_cores,
        memory_gib=memory_gib,
        max_usd=max_usd,
        max_gpu_hours=max_gpu_hours,
        retries=retries,
    )
    limit = raw.get("limit")
    return TrainPlan(
        gpu=gpu,
        seconds=seconds,
        cpu_cores=cpu_cores,
        memory_gib=memory_gib,
        max_usd=max_usd,
        max_gpu_hours=max_gpu_hours,
        retries=retries,
        limit=None if limit is None else int(limit),
        epochs=int(raw["epochs"]),
        estimate_usd=usd,
        start_step=_start_step(checkpoint_dir),
        volume_name=str(raw["volume"]),
        checkpoint_every=int(raw["checkpoint_every"]),
    )


def function_spec(plan: TrainPlan) -> dict:
    return {
        "gpu": plan.gpu,
        "cpu": plan.cpu_cores,
        "memory": plan.memory_gib * 1024,
        "timeout": plan.seconds,
        "retries": plan.retries,
        "volumes": {"/checkpoints": plan.volume_name},
    }


def record_checkpoint(directory: str | Path, step: int) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / "latest.json").write_text(
        json.dumps({"step": step}) + "\n",
        encoding="utf-8",
    )


def train(config: str = "configs/modal-lean.yaml") -> TrainPlan:
    """Print the estimate and refuse a ceiling breach. Does not boot a GPU."""
    return plan_train(config)


def execute_train(
    config_text: str,
    pack_text: str,
    checkpoint_dir: str | Path,
    *,
    archive: bytes | None = None,
    session=None,
    on_checkpoint=None,
):
    """Guard the ceiling, fit prepared pairs, and save the LoRA adapter."""
    import tempfile

    from rosai.cuad import load_categories
    from rosai.pack import load_pack
    from rosai.prepare import prepare
    from rosai.train import Manifest, TrainingError, UnslothBackend, load_run
    from rosai.train import train as write_run

    destination = Path(checkpoint_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "run.yaml"
        config_path.write_text(config_text, encoding="utf-8")
        plan = plan_train(config_path, checkpoint_dir=destination)
        run = load_run(config_path)
        if run.backend != "unsloth":
            raise TrainingError(f"backend {run.backend!r} is not the QLoRA trainer")
        categories = load_categories(config_path)
        pack_path = Path(tmp) / "pack.yaml"
        pack_path.write_text(pack_text, encoding="utf-8")
        prepared = prepare(
            load_pack(pack_path),
            archive=archive,
            categories=categories,
            max_seq_length=run.max_seq_length,
        )
    backend = UnslothBackend(
        run,
        destination,
        epochs=plan.epochs,
        checkpoint_every=plan.checkpoint_every,
        start_step=plan.start_step,
        session=session,
        on_checkpoint=on_checkpoint,
    )
    manifest: Manifest = write_run(prepared, run, backend, destination)
    record_checkpoint(destination, backend.step)
    return manifest


@dataclass(frozen=True)
class EvalPlan:
    gpu: str
    seconds: int
    cpu_cores: int
    memory_gib: int
    max_usd: float
    max_gpu_hours: float
    retries: int
    limit: int | None
    batch_size: int
    estimate_usd: float
    capped: bool
    volume_name: str
    model_id: str


def plan_eval(path: str | Path) -> EvalPlan:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    seconds = int(raw["gpu_seconds"])
    cpu_cores = int(raw["cpu_cores"])
    memory_gib = int(raw["memory_gib"])
    gpu = str(raw["gpu"])
    max_usd = float(raw["max_usd"])
    max_gpu_hours = float(raw["max_gpu_hours"])
    retries = _retry_count(raw)
    usd = guard(
        gpu=gpu,
        seconds=seconds,
        cpu_cores=cpu_cores,
        memory_gib=memory_gib,
        max_usd=max_usd,
        max_gpu_hours=max_gpu_hours,
        retries=retries,
    )
    limit = raw.get("limit")
    limit_value = None if limit is None else int(limit)
    return EvalPlan(
        gpu=gpu,
        seconds=seconds,
        cpu_cores=cpu_cores,
        memory_gib=memory_gib,
        max_usd=max_usd,
        max_gpu_hours=max_gpu_hours,
        retries=retries,
        limit=limit_value,
        batch_size=int(raw["batch_size"]),
        estimate_usd=usd,
        capped=limit_value is not None,
        volume_name=str(raw["volume"]),
        model_id=str(raw["model"]["id"]),
    )


def evaluate(config: str = "configs/modal-eval-lean.yaml") -> EvalPlan:
    """Print the eval estimate and refuse a ceiling breach. Does not boot a GPU."""
    return plan_eval(config)


class EvalError(Exception):
    """Held-out generation cannot run as configured."""


def execute_eval(
    config_text: str,
    pack_text: str,
    checkpoint_dir: str | Path,
    *,
    archive: bytes | None = None,
    engine=None,
) -> int:
    """Guard the ceiling, generate held-out answers, and write predictions."""
    import tempfile

    from rosai.cuad import load_categories
    from rosai.eval_batch import clean_answer, run_batches, write_predictions
    from rosai.pack import load_pack
    from rosai.prepare import prepare
    from rosai.train import TrainingError, format_prompt

    destination = Path(checkpoint_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "run.yaml"
        config_path.write_text(config_text, encoding="utf-8")
        plan = plan_eval(config_path)
        raw = yaml.safe_load(config_text)
        model = raw["model"]
        if bool(model["thinking"]):
            raise TrainingError("Qwen3 QLoRA keeps thinking off")
        max_seq_length = int(model.get("max_seq_length", 2048))
        categories = load_categories(config_path)
        pack_path = Path(tmp) / "pack.yaml"
        pack_path.write_text(pack_text, encoding="utf-8")
        prepared = prepare(
            load_pack(pack_path),
            archive=archive,
            categories=categories,
            max_seq_length=max_seq_length,
        )
    examples = prepared.held_out
    if plan.limit is not None:
        examples = examples[: plan.limit]
    prompts = [format_prompt(pair, thinking=False) for pair in examples]
    if engine is None:
        adapter_dir = destination / "adapter"
        if not (adapter_dir / "adapter_model.safetensors").is_file():
            raise EvalError(f"no LoRA adapter in {adapter_dir}")
        from rosai.eval_batch import VllmEngine

        engine = VllmEngine(plan.model_id, adapter_dir=str(adapter_dir))
    completions = run_batches(engine, prompts, plan.batch_size)
    rows = [
        {
            "document_id": pair.document_id,
            "user": pair.messages[0][1],
            "prediction": clean_answer(prediction),
        }
        for pair, prediction in zip(examples, completions, strict=True)
    ]
    write_predictions(destination, rows)
    return len(rows)


def _retry_count(raw: dict) -> int:
    """Retries after the first attempt. Omitted means one, the lean default."""
    return int(raw.get("retries", 1))


def _start_step(checkpoint_dir: str | Path | None) -> int:
    if checkpoint_dir is None:
        return 0
    marker = Path(checkpoint_dir) / "latest.json"
    if not marker.is_file():
        return 0
    return int(json.loads(marker.read_text(encoding="utf-8"))["step"])


def _deployed_resources(volume, modal) -> dict:
    config = Path("configs/modal-full.yaml")
    if config.is_file():
        raw = yaml.safe_load(config.read_text(encoding="utf-8"))
        gpu = str(raw["gpu"])
        cpu = int(raw["cpu_cores"])
        memory = int(raw["memory_gib"]) * 1024
        timeout = int(raw["gpu_seconds"])
        retries = _retry_count(raw)
    else:
        gpu, cpu, memory, timeout, retries = "A100-80GB", 8, 32768, 10800, 1
    return {
        "gpu": gpu,
        "cpu": cpu,
        "memory": memory,
        "timeout": timeout,
        "volumes": {"/checkpoints": volume},
        "secrets": [modal.Secret.from_name("huggingface")],
        "retries": modal.Retries(initial_delay=0.0, max_retries=retries),
    }


def _eval_resources(volume, modal) -> dict:
    config = Path("configs/modal-eval-full.yaml")
    if config.is_file():
        raw = yaml.safe_load(config.read_text(encoding="utf-8"))
        gpu = str(raw["gpu"])
        cpu = int(raw["cpu_cores"])
        memory = int(raw["memory_gib"]) * 1024
        timeout = int(raw["gpu_seconds"])
        retries = _retry_count(raw)
    else:
        gpu, cpu, memory, timeout, retries = "A100-80GB", 8, 32768, 1800, 1
    return {
        "gpu": gpu,
        "cpu": cpu,
        "memory": memory,
        "timeout": timeout,
        "volumes": {"/checkpoints": volume},
        "secrets": [modal.Secret.from_name("huggingface")],
        "retries": modal.Retries(initial_delay=0.0, max_retries=retries),
    }


# CUDA 12.8 wheels cover the preemptible A100. Importing this module does not import them.
_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
_TRAIN_PACKAGES = (
    "pyyaml",
    "unsloth",
    "trl",
    "transformers",
    "datasets",
    "peft",
    "bitsandbytes",
    "accelerate",
    "hf_transfer",
)


def _train_image(modal):
    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch", extra_index_url=_CUDA_INDEX)
        .pip_install(*_TRAIN_PACKAGES)
        .env({"HF_HOME": "/checkpoints/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .add_local_python_source("rosai")
    )


def _attach(modal) -> None:
    """Decorate the remote functions only when the Modal SDK is installed."""
    global train, evaluate, app
    volume = modal.Volume.from_name("rosai-checkpoints", create_if_missing=True)
    image = modal.Image.debian_slim(python_version="3.12").pip_install("pyyaml")
    app = modal.App("rosai-train", image=image)
    resources = _deployed_resources(volume, modal)
    resources["image"] = _train_image(modal)

    @app.function(**resources)
    def remote_train(config_text: str, pack_text: str, session=None) -> int:
        root = Path(os.environ.get("ROSAI_CHECKPOINT_DIR", "/checkpoints"))
        previous = os.environ.get("ROSAI_DOWNLOAD_CUAD")
        os.environ["ROSAI_DOWNLOAD_CUAD"] = "1"
        try:
            manifest = execute_train(
                config_text,
                pack_text,
                root,
                session=session,
                on_checkpoint=volume.commit,
            )
        finally:
            if previous is None:
                os.environ.pop("ROSAI_DOWNLOAD_CUAD", None)
            else:
                os.environ["ROSAI_DOWNLOAD_CUAD"] = previous
        volume.commit()
        return manifest.example_count

    @app.local_entrypoint()
    def train(
        config: str = "configs/modal-lean.yaml",
        pack: str = "packs/legal.yaml",
    ) -> None:
        plan = plan_train(config)
        remote_train.with_options(
            timeout=plan.seconds,
            retries=modal.Retries(initial_delay=0.0, max_retries=plan.retries),
        ).remote(
            Path(config).read_text(encoding="utf-8"),
            Path(pack).read_text(encoding="utf-8"),
        )

    eval_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("pyyaml", "vllm")
        .env({"HF_HOME": "/checkpoints/hf"})
        .add_local_python_source("rosai")
    )
    eval_resources = _eval_resources(volume, modal)
    eval_resources["image"] = eval_image

    @app.function(**eval_resources)
    def remote_eval(config_text: str, pack_text: str, engine=None) -> int:
        root = Path(os.environ.get("ROSAI_CHECKPOINT_DIR", "/checkpoints"))
        previous = os.environ.get("ROSAI_DOWNLOAD_CUAD")
        os.environ["ROSAI_DOWNLOAD_CUAD"] = "1"
        try:
            count = execute_eval(
                config_text,
                pack_text,
                root,
                engine=engine,
            )
        finally:
            if previous is None:
                os.environ.pop("ROSAI_DOWNLOAD_CUAD", None)
            else:
                os.environ["ROSAI_DOWNLOAD_CUAD"] = previous
        volume.commit()
        return count

    @app.local_entrypoint()
    def evaluate(
        config: str = "configs/modal-eval-lean.yaml",
        pack: str = "packs/legal.yaml",
    ) -> None:
        plan = plan_eval(config)
        remote_eval.with_options(
            timeout=plan.seconds,
            retries=modal.Retries(initial_delay=0.0, max_retries=plan.retries),
        ).remote(
            Path(config).read_text(encoding="utf-8"),
            Path(pack).read_text(encoding="utf-8"),
        )


app = None

try:
    import modal as _modal
except ImportError:
    _modal = None

if _modal is not None:
    _attach(_modal)

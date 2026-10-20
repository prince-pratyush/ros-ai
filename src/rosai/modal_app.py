"""Preemptible Modal train plan. The GPU stays off until modal run."""

import json
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
    usd = guard(
        gpu=gpu,
        seconds=seconds,
        cpu_cores=cpu_cores,
        memory_gib=memory_gib,
        max_usd=max_usd,
        max_gpu_hours=max_gpu_hours,
    )
    limit = raw.get("limit")
    return TrainPlan(
        gpu=gpu,
        seconds=seconds,
        cpu_cores=cpu_cores,
        memory_gib=memory_gib,
        max_usd=max_usd,
        max_gpu_hours=max_gpu_hours,
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
    from rosai.train import Manifest, TrainingError, UnslothBackend, load_run, train

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
    manifest: Manifest = train(prepared, run, backend, destination)
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
    usd = guard(
        gpu=gpu,
        seconds=seconds,
        cpu_cores=cpu_cores,
        memory_gib=memory_gib,
        max_usd=max_usd,
        max_gpu_hours=max_gpu_hours,
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
    else:
        gpu, cpu, memory, timeout = "A100-80GB", 8, 32768, 10800
    return {
        "gpu": gpu,
        "cpu": cpu,
        "memory": memory,
        "timeout": timeout,
        "volumes": {"/checkpoints": volume},
        "retries": modal.Retries(initial_delay=0.0, max_retries=3),
    }


def _eval_resources(volume, modal) -> dict:
    config = Path("configs/modal-eval-full.yaml")
    if config.is_file():
        raw = yaml.safe_load(config.read_text(encoding="utf-8"))
        gpu = str(raw["gpu"])
        cpu = int(raw["cpu_cores"])
        memory = int(raw["memory_gib"]) * 1024
        timeout = int(raw["gpu_seconds"])
    else:
        gpu, cpu, memory, timeout = "A100-80GB", 8, 32768, 1800
    return {
        "gpu": gpu,
        "cpu": cpu,
        "memory": memory,
        "timeout": timeout,
        "volumes": {"/checkpoints": volume},
        "retries": modal.Retries(initial_delay=0.0, max_retries=3),
    }


def _attach(modal) -> None:
    """Decorate the remote functions only when the Modal SDK is installed."""
    global train, evaluate
    volume = modal.Volume.from_name("rosai-checkpoints", create_if_missing=True)
    image = modal.Image.debian_slim(python_version="3.12").pip_install("pyyaml")
    app = modal.App("rosai-train", image=image)
    resources = _deployed_resources(volume, modal)

    @app.function(**resources)
    def remote_train(config_text: str) -> int:
        destination = Path("/tmp/rosai-run.yaml")
        destination.write_text(config_text, encoding="utf-8")
        plan = plan_train(destination, checkpoint_dir=Path("/checkpoints"))
        record_checkpoint(Path("/checkpoints"), plan.start_step)
        volume.commit()
        return plan.start_step

    @app.local_entrypoint()
    def train(config: str = "configs/modal-lean.yaml") -> None:
        plan_train(config)
        remote_train.remote(Path(config).read_text(encoding="utf-8"))

    eval_image = modal.Image.debian_slim(python_version="3.12").pip_install(
        "pyyaml", "vllm"
    )
    eval_resources = _eval_resources(volume, modal)
    eval_resources["image"] = eval_image

    @app.function(**eval_resources)
    def remote_eval(config_text: str) -> int:
        destination = Path("/tmp/rosai-eval.yaml")
        destination.write_text(config_text, encoding="utf-8")
        plan = plan_eval(destination)
        prompts_path = Path("/checkpoints/prompts.json")
        if not prompts_path.is_file():
            volume.commit()
            return 0
        from rosai.eval_batch import VllmEngine, run_batches, write_completions

        prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
        completions = run_batches(VllmEngine(plan.model_id), prompts, plan.batch_size)
        write_completions(Path("/checkpoints"), completions)
        volume.commit()
        return len(completions)

    @app.local_entrypoint()
    def evaluate(config: str = "configs/modal-eval-lean.yaml") -> None:
        plan_eval(config)
        remote_eval.remote(Path(config).read_text(encoding="utf-8"))


try:
    import modal as _modal
except ImportError:
    _modal = None

if _modal is not None:
    _attach(_modal)

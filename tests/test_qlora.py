import json
import sys
from pathlib import Path

import pytest

from rosai.pack import load_pack
from rosai.prepare import InstructionPair, prepare

# Rendered from Qwen/Qwen3-8B tokenizer_config.json chat_template
# with add_generation_prompt=False. The empty think block is the
# non-thinking assistant turn.
_THINKING_OFF = (
    "<|im_start|>user\n"
    "Quote the Governing Law clause or answer none.<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
    "Nevada<|im_end|>\n"
)


def test_prepared_pair_formats_as_qwen3_with_thinking_off():
    pair = InstructionPair(
        document_id="contract-a",
        task="clause_span",
        messages=(("user", "Quote the Governing Law clause or answer none."),),
        completion="Nevada",
    )

    from rosai.train import format_pair

    assert format_pair(pair, thinking=False) == _THINKING_OFF


def test_thinking_on_is_refused():
    pair = InstructionPair(
        document_id="contract-a",
        task="clause_span",
        messages=(("user", "Quote the Governing Law clause or answer none."),),
        completion="Nevada",
    )

    from rosai.train import TrainingError, format_pair

    with pytest.raises(TrainingError):
        format_pair(pair, thinking=True)


# Qwen target modules from the Unsloth QLoRA recipe.
_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class _SavingSession:
    def __init__(self) -> None:
        self.request = None
        self.on_checkpoint = None

    def fit(self, request, on_checkpoint=None) -> int:
        self.request = request
        self.on_checkpoint = on_checkpoint
        if on_checkpoint is not None:
            on_checkpoint()
        path = Path(request.adapter_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (path / "adapter_model.safetensors").write_bytes(b"lora")
        return 4


def test_unsloth_backend_fits_qwen3_qlora_and_saves_the_adapter(tmp_path):
    from rosai.train import UnslothBackend, load_run, train

    prepared = prepare(load_pack(Path("tests/fixtures/cuad_mini/pack.yaml")))
    run = load_run(Path("configs/modal-lean.yaml"))
    session = _SavingSession()
    backend = UnslothBackend(
        run,
        tmp_path,
        epochs=1,
        checkpoint_every=50,
        session=session,
    )

    manifest = train(prepared, run, backend, tmp_path)

    request = session.request
    assert request.texts == (_THINKING_OFF,)
    assert request.model_id == "Qwen/Qwen3-8B"
    assert request.load_in_4bit is True
    assert request.completion_only is True
    assert request.target_modules == _TARGETS
    assert request.lora_r == 16
    assert request.lora_alpha == 16
    assert request.seed == 3407
    assert request.max_seq_length == 2048
    assert request.batch_size == 1
    assert request.resume_step == 0
    assert (tmp_path / "adapter" / "adapter_model.safetensors").read_bytes() == b"lora"
    assert manifest.example_count == 1
    assert manifest.backend == "unsloth"
    assert manifest.thinking is False
    assert "unsloth" not in sys.modules
    assert "torch" not in sys.modules


class _SilentSession:
    def fit(self, request, on_checkpoint=None) -> int:
        return 0


def test_fit_fails_when_the_session_does_not_save_the_adapter(tmp_path):
    from rosai.train import TrainingError, UnslothBackend, load_run, train

    prepared = prepare(load_pack(Path("tests/fixtures/cuad_mini/pack.yaml")))
    run = load_run(Path("configs/modal-lean.yaml"))
    backend = UnslothBackend(
        run,
        tmp_path,
        epochs=1,
        checkpoint_every=50,
        session=_SilentSession(),
    )

    with pytest.raises(TrainingError):
        train(prepared, run, backend, tmp_path)

    assert not (tmp_path / "manifest.json").exists()


def test_a_checkpoint_save_records_the_step_and_commits(tmp_path):
    from rosai.train import on_save_checkpoint

    commits: list[str] = []
    on_save_checkpoint(tmp_path, 50, lambda: commits.append("commit"))

    written = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert written == {"step": 50}
    assert commits == ["commit"]
    assert "unsloth" not in sys.modules
    assert "torch" not in sys.modules
    assert "modal" not in sys.modules


def _clause_pack(tmp_path: Path) -> str:
    (tmp_path / "labels.csv").write_text(
        "document_id,split,Governing Law,Parties\n"
        "contract-a,train,Nevada,Acme\n"
        "contract-c,train,Delaware,Gamma\n"
        "contract-b,test,New York,Beta\n",
        encoding="utf-8",
    )
    return (
        "id: mini\n"
        "dev_fraction: 0\n"
        "sources:\n"
        "  - id: mini\n"
        "    licence: cc-by-4.0\n"
        f"    path: {tmp_path}\n"
        "attribution:\n"
        '  - "The Atticus Project, CUAD v1, CC BY 4.0"\n'
    )


def test_execute_train_fits_the_named_category_under_the_cost_guard(tmp_path, capsys):
    from rosai.modal_app import execute_train

    config = (
        Path("configs/modal-lean.yaml")
        .read_text(encoding="utf-8")
        .replace("limit: 256", "limit: 1")
    )
    session = _SavingSession()
    commits: list[int] = []
    volume = tmp_path / "volume"

    manifest = execute_train(
        config,
        _clause_pack(tmp_path),
        volume,
        session=session,
        on_checkpoint=lambda: commits.append(1),
    )

    assert session.request.texts == (_THINKING_OFF,)
    assert session.request.epochs == 1
    assert session.request.checkpoint_every == 50
    assert session.request.resume_step == 0
    assert session.request.load_in_4bit is True
    assert manifest.example_count == 1
    assert manifest.model_id == "Qwen/Qwen3-8B"
    assert manifest.thinking is False
    assert manifest.backend == "unsloth"
    assert (volume / "adapter" / "adapter_model.safetensors").read_bytes() == b"lora"
    assert json.loads((volume / "latest.json").read_text(encoding="utf-8")) == {
        "step": 4
    }
    assert commits == [1]
    assert "1.043808" in capsys.readouterr().out
    assert "unsloth" not in sys.modules
    assert "torch" not in sys.modules


def test_execute_train_refuses_a_ceiling_breach_before_fit(tmp_path):
    from rosai.cost import CostCeilingError
    from rosai.modal_app import execute_train

    config = (
        Path("configs/modal-lean.yaml")
        .read_text(encoding="utf-8")
        .replace("gpu_seconds: 1200", "gpu_seconds: 3600")
    )
    session = _SavingSession()

    with pytest.raises(CostCeilingError):
        execute_train(
            config,
            _clause_pack(tmp_path),
            tmp_path / "volume",
            session=session,
        )

    assert session.request is None
    assert not (tmp_path / "volume" / "adapter").exists()


def test_execute_train_resumes_from_the_recorded_step(tmp_path):
    from rosai.modal_app import execute_train, record_checkpoint

    volume = tmp_path / "volume"
    record_checkpoint(volume, 50)
    session = _SavingSession()

    execute_train(
        Path("configs/modal-lean.yaml").read_text(encoding="utf-8"),
        _clause_pack(tmp_path),
        volume,
        session=session,
    )

    assert session.request.resume_step == 50
    assert json.loads((volume / "latest.json").read_text(encoding="utf-8")) == {
        "step": 4
    }


def test_execute_train_refuses_a_backend_other_than_unsloth(tmp_path):
    from rosai.modal_app import execute_train
    from rosai.train import TrainingError

    config = (
        Path("configs/modal-lean.yaml")
        .read_text(encoding="utf-8")
        .replace("backend: unsloth", "backend: fake")
    )
    session = _SavingSession()

    with pytest.raises(TrainingError):
        execute_train(
            config,
            _clause_pack(tmp_path),
            tmp_path / "volume",
            session=session,
        )

    assert session.request is None


def test_default_backend_imports_unsloth_only_inside_fit(tmp_path):
    from rosai.train import UnslothBackend, load_run, train

    prepared = prepare(load_pack(Path("tests/fixtures/cuad_mini/pack.yaml")))
    run = load_run(Path("configs/modal-lean.yaml"))
    backend = UnslothBackend(run, tmp_path, epochs=1, checkpoint_every=50)

    with pytest.raises(ImportError):
        train(prepared, run, backend, tmp_path)

    assert not (tmp_path / "manifest.json").exists()
    assert "unsloth" not in sys.modules
    assert "torch" not in sys.modules

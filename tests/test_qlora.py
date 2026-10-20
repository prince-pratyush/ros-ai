import json
import os
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


def test_resume_uses_the_latest_trainer_checkpoint(tmp_path):
    from rosai.train import resume_checkpoint

    assert resume_checkpoint(str(tmp_path), 0) is False
    assert resume_checkpoint(str(tmp_path), 50) is False
    (tmp_path / "checkpoint-20").mkdir()
    (tmp_path / "checkpoint-100").mkdir()

    assert resume_checkpoint(str(tmp_path), 50) == str(tmp_path / "checkpoint-100")


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


def test_execute_train_windows_the_contract_to_the_configured_length(tmp_path):
    import io
    import json
    import zipfile

    from rosai.modal_app import execute_train

    context = ("A" * 16) + ("B" * 16)
    qas = [
        {
            "id": "contract-a__Governing Law_0",
            "question": (
                'Highlight the parts (if any) of this contract related to '
                '"Governing Law" that should be reviewed by a lawyer.'
            ),
            "answers": [{"text": "B" * 16, "answer_start": 16}],
            "is_impossible": False,
        }
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "train_separate_questions.json",
            json.dumps(
                {
                    "version": "1",
                    "data": [
                        {
                            "title": "contract-a",
                            "paragraphs": [{"context": context, "qas": qas}],
                        }
                    ],
                }
            ),
        )
        archive.writestr("test.json", json.dumps({"version": "1", "data": []}))
    config = (
        Path("configs/modal-lean.yaml")
        .read_text(encoding="utf-8")
        .replace("max_seq_length: 2048", "max_seq_length: 40")
        .replace("limit: 256", "limit: 8")
    )
    session = _SavingSession()

    execute_train(
        config,
        "id: mini\ndev_fraction: 0\n",
        tmp_path / "volume",
        archive=buffer.getvalue(),
        session=session,
    )

    question = "Quote the Governing Law clause or answer none."
    assert session.request.texts[0].startswith(
        f"<|im_start|>user\n{'A' * 16}\n\n{question}<|im_end|>\n"
    )
    assert session.request.texts[0].endswith("none<|im_end|>\n")
    assert session.request.texts[1].startswith(
        f"<|im_start|>user\n{'B' * 16}\n\n{question}<|im_end|>\n"
    )
    assert session.request.texts[1].endswith(f"{'B' * 16}<|im_end|>\n")
    assert len(session.request.texts) == 2
    assert session.request.max_seq_length == 40


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


def test_execute_train_refuses_an_empty_train_split(tmp_path):
    from rosai.modal_app import execute_train
    from rosai.train import TrainingError

    (tmp_path / "labels.csv").write_text(
        "document_id,split,Governing Law\ncontract-b,test,Nevada\n",
        encoding="utf-8",
    )
    pack = (
        "id: mini\n"
        "dev_fraction: 0\n"
        "sources:\n"
        "  - id: mini\n"
        "    licence: cc-by-4.0\n"
        f"    path: {tmp_path}\n"
    )
    session = _SavingSession()

    with pytest.raises(TrainingError):
        execute_train(
            Path("configs/modal-lean.yaml").read_text(encoding="utf-8"),
            pack,
            tmp_path / "volume",
            session=session,
        )

    assert session.request is None
    assert not (tmp_path / "volume" / "adapter").exists()


def test_execute_train_refuses_thinking(tmp_path):
    from rosai.modal_app import execute_train
    from rosai.train import TrainingError

    config = (
        Path("configs/modal-lean.yaml")
        .read_text(encoding="utf-8")
        .replace("thinking: false", "thinking: true")
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


class _Image:
    def __init__(self) -> None:
        self.packages: list[str] = []
        self.indexes: list[str] = []
        self.sources: tuple[str, ...] = ()
        self.variables: dict[str, str] = {}

    def pip_install(self, *packages: str, **kwargs):
        self.packages.extend(packages)
        index = kwargs.get("extra_index_url")
        if index:
            self.indexes.append(index)
        return self

    def env(self, values: dict[str, str]):
        self.variables.update(values)
        return self

    def add_local_python_source(self, *names: str):
        self.sources = names
        return self


class _Volume:
    def __init__(self, name: str) -> None:
        self.name = name
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1


class _FakeModal:
    """Stand-in for the Modal SDK. It records the function and does not boot a GPU."""

    def __init__(self) -> None:
        self.functions: dict = {}
        self.volume: _Volume | None = None
        owner = self

        class Image:
            @staticmethod
            def debian_slim(python_version: str = "3.12") -> _Image:
                return _Image()

        class Volume:
            @staticmethod
            def from_name(name: str, create_if_missing: bool = False) -> _Volume:
                owner.volume = _Volume(name)
                return owner.volume

        class Retries:
            def __init__(self, initial_delay: float = 0.0, max_retries: int = 0) -> None:
                self.initial_delay = initial_delay
                self.max_retries = max_retries

        class App:
            def __init__(self, name: str, image=None) -> None:
                self.name = name
                self.image = image

            def function(self, **resources):
                def deco(fn):
                    remote = _Remote(fn, resources)
                    owner.functions[fn.__name__] = remote
                    return remote

                return deco

            def local_entrypoint(self):
                def deco(fn):
                    return fn

                return deco

        self.Image = Image
        self.Volume = Volume
        self.Retries = Retries
        self.App = App


class _Remote:
    def __init__(self, fn, resources: dict) -> None:
        self.fn = fn
        self.resources = resources
        self.calls: list[tuple] = []

    def remote(self, *args, **kwargs):
        self.calls.append(args)
        return None


def test_remote_train_saves_the_lora_adapter_and_commits_the_volume(
    tmp_path, monkeypatch
):
    import rosai.modal_app as app

    monkeypatch.setenv("ROSAI_CHECKPOINT_DIR", str(tmp_path / "vol"))
    fake = _FakeModal()
    previous = (app.train, app.evaluate)
    try:
        app._attach(fake)
        remote = fake.functions["remote_train"]
        assert remote.resources["gpu"] == "A100-80GB"
        assert "nonpreemptible" not in remote.resources
        assert remote.resources["cpu"] == 8
        assert remote.resources["memory"] == 32768
        assert remote.resources["timeout"] == 10800
        assert remote.resources["volumes"]["/checkpoints"].name == "rosai-checkpoints"
        assert remote.resources["retries"].max_retries == 3
        image = remote.resources["image"]
        assert "unsloth" in image.packages
        assert "torch" in image.packages
        assert image.indexes == ["https://download.pytorch.org/whl/cu128"]
        assert image.variables["HF_HOME"] == "/checkpoints/hf"
        assert "rosai" in image.sources
        session = _SavingSession()
        count = remote.fn(
            Path("configs/modal-lean.yaml").read_text(encoding="utf-8"),
            _clause_pack(tmp_path),
            session,
        )
    finally:
        app.train, app.evaluate = previous

    assert count == 2
    assert session.request.texts[0] == _THINKING_OFF
    assert session.request.texts[1].endswith("Delaware<|im_end|>\n")
    assert "Acme" not in "".join(session.request.texts)
    assert "New York" not in "".join(session.request.texts)
    assert session.request.load_in_4bit is True
    assert session.request.model_id == "Qwen/Qwen3-8B"
    volume = tmp_path / "vol"
    assert (volume / "adapter" / "adapter_model.safetensors").read_bytes() == b"lora"
    written = json.loads((volume / "manifest.json").read_text(encoding="utf-8"))
    assert written["example_count"] == 2
    assert written["backend"] == "unsloth"
    assert written["thinking"] is False
    assert fake.volume is not None
    assert fake.volume.commits == 2
    assert "ROSAI_DOWNLOAD_CUAD" not in os.environ
    assert "unsloth" not in sys.modules
    assert "torch" not in sys.modules
    assert "modal" not in sys.modules


def test_remote_train_does_not_commit_a_refused_run(tmp_path, monkeypatch):
    import rosai.modal_app as app
    from rosai.cost import CostCeilingError

    monkeypatch.setenv("ROSAI_CHECKPOINT_DIR", str(tmp_path / "vol"))
    fake = _FakeModal()
    previous = (app.train, app.evaluate)
    config = (
        Path("configs/modal-lean.yaml")
        .read_text(encoding="utf-8")
        .replace("gpu_seconds: 1200", "gpu_seconds: 3600")
    )
    session = _SavingSession()
    try:
        app._attach(fake)
        with pytest.raises(CostCeilingError):
            fake.functions["remote_train"].fn(
                config, _clause_pack(tmp_path), session
            )
    finally:
        app.train, app.evaluate = previous

    assert session.request is None
    assert fake.volume is not None
    assert fake.volume.commits == 0
    assert not (tmp_path / "vol" / "adapter").exists()


def test_entrypoint_sends_the_pack_and_refuses_before_a_remote_call(tmp_path):
    import rosai.modal_app as app
    from rosai.cost import CostCeilingError

    fake = _FakeModal()
    previous = (app.train, app.evaluate)
    pack = tmp_path / "pack.yaml"
    pack.write_text("id: local\n", encoding="utf-8")
    try:
        app._attach(fake)
        app.train("configs/modal-lean.yaml", str(pack))
        sent = fake.functions["remote_train"].calls
        assert sent == [
            (
                Path("configs/modal-lean.yaml").read_text(encoding="utf-8"),
                "id: local\n",
            )
        ]
        refused = tmp_path / "refused.yaml"
        refused.write_text(
            Path("configs/modal-lean.yaml")
            .read_text(encoding="utf-8")
            .replace("gpu_seconds: 1200", "gpu_seconds: 3600"),
            encoding="utf-8",
        )
        with pytest.raises(CostCeilingError):
            app.train(str(refused), str(pack))
    finally:
        app.train, app.evaluate = previous

    assert len(fake.functions["remote_train"].calls) == 1


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

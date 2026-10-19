import json
from pathlib import Path

from rosai.pack import load_pack
from rosai.prepare import prepare
from rosai.train import load_run, train

# sha256 of configs/fake.yaml, computed outside this test.
CONFIG_HASH = "2cb5fe3ce187ead9ea4bc1207187fa5b2d5fd9af8eed957e7673ae2bb81a66e8"


class _RecordingBackend:
    def __init__(self) -> None:
        self.document_ids: list[str] = []
        self.completion_only: bool | None = None

    def fit(self, examples, *, completion_only: bool) -> None:
        self.document_ids.extend(example.document_id for example in examples)
        self.completion_only = completion_only


def test_train_writes_a_manifest_for_the_fake_backend(tmp_path):
    prepared = prepare(load_pack(Path("tests/fixtures/cuad_mini/pack.yaml")))
    run = load_run(Path("configs/fake.yaml"))
    backend = _RecordingBackend()

    manifest = train(prepared, run, backend, tmp_path)

    written = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert written["model_id"] == "Qwen/Qwen3-8B"
    assert written["thinking"] is False
    assert written["seed"] == 3407
    assert written["loss"] == "completion"
    assert written["config_hash"] == CONFIG_HASH
    assert manifest.model_id == written["model_id"]
    assert manifest.thinking is written["thinking"]
    assert manifest.seed == written["seed"]
    assert manifest.loss == written["loss"]
    assert manifest.config_hash == written["config_hash"]
    assert backend.completion_only is True


def test_trainer_does_not_receive_held_out_ids(tmp_path):
    prepared = prepare(load_pack(Path("tests/fixtures/cuad_published/pack.yaml")))
    backend = _RecordingBackend()

    train(prepared, load_run(Path("configs/fake.yaml")), backend, tmp_path)

    assert set(backend.document_ids) == {
        "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT",
        "NELNETINC_04_08_2020-EX-1-JOINT FILING AGREEMENT",
    }

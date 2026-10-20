import json
import sys
from pathlib import Path

import pytest

from rosai.__main__ import main

# gold tokens: nevada. prediction tokens: nevada, law.
# overlap 1, precision 1/2, recall 1, F1 = 2 * 0.5 * 1 / 1.5.
NEVADA_LAW_F1 = 0.6666666666666666


def test_prepare_uses_categories_and_the_config_window(tmp_path, capsys):
    pack = _pack(tmp_path, ("A" * 16) + "Nevada" + ("B" * 10))
    config = tmp_path / "run.yaml"
    config.write_text(
        "model:\n"
        "  id: Qwen/Qwen3-8B\n"
        "  thinking: false\n"
        "  max_seq_length: 40\n"
        "categories:\n"
        "  - Governing Law\n"
        "  - Parties\n",
        encoding="utf-8",
    )

    main(["prepare", "--pack", str(pack), "--config", str(config)])

    # Two categories, and the 32-character contract is two 16-character windows.
    assert capsys.readouterr().out.strip() == "train 4 dev 0 held_out 0"

    main(
        [
            "prepare",
            "--pack",
            str(pack),
            "--config",
            str(config),
            "--categories",
            "Parties",
        ]
    )
    assert capsys.readouterr().out.strip() == "train 2 dev 0 held_out 0"


def test_train_fits_each_contract_window(tmp_path):
    pack = _pack(tmp_path, ("A" * 16) + "Nevada" + ("B" * 10))
    config = tmp_path / "run.yaml"
    config.write_text(
        Path("configs/mac-dry.yaml")
        .read_text(encoding="utf-8")
        .replace("max_seq_length: 1024", "max_seq_length: 40"),
        encoding="utf-8",
    )
    dest = tmp_path / "run"

    main(
        [
            "train",
            "--pack",
            str(pack),
            "--config",
            str(config),
            "--dest",
            str(dest),
        ]
    )

    written = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert written["example_count"] == 2
    assert written["max_seq_length"] == 40
    assert "modal" not in sys.modules
    assert "unsloth" not in sys.modules


def test_score_writes_exact_match_and_token_f1(tmp_path):
    pack = _held_out_pack(tmp_path)
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            [
                {
                    "document_id": "held-span",
                    "user": "Quote the Governing Law clause or answer none.",
                    "prediction": "Nevada law",
                }
            ]
        ),
        encoding="utf-8",
    )
    dest = tmp_path / "score"
    config = tmp_path / "eval.yaml"
    config.write_text(
        "model:\n  id: Qwen/Qwen3-8B\n  thinking: false\nlimit: 32\n",
        encoding="utf-8",
    )

    main(
        [
            "score",
            "--pack",
            str(pack),
            "--predictions",
            str(predictions),
            "--dest",
            str(dest),
            "--config",
            str(config),
        ]
    )

    card = json.loads((dest / "scorecard.json").read_text(encoding="utf-8"))
    tuned = card["candidates"]["tuned"]
    assert tuned["exact_match"] == 0
    assert tuned["token_f1"] == NEVADA_LAW_F1
    assert tuned["support"] == 1
    assert card["capped"] is True
    assert card["traces_sent"] == 0
    assert "hosted" not in card
    assert "vllm" not in sys.modules
    assert "modal" not in sys.modules
    assert "langsmith" not in sys.modules


def test_upload_langsmith_needs_a_key_and_stops_at_five_thousand(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    pack, predictions = _many_held_out(tmp_path, 3)
    dest = tmp_path / "plain"

    main(
        [
            "score",
            "--pack",
            str(pack),
            "--predictions",
            str(predictions),
            "--dest",
            str(dest),
            "--upload-langsmith",
        ]
    )

    card = json.loads((dest / "scorecard.json").read_text(encoding="utf-8"))
    assert card["traces_sent"] == 0
    assert "langsmith" not in sys.modules

    created: list = []

    class _Client:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.runs: list[dict] = []
            created.append(self)

        def create_run(self, **run) -> None:
            self.runs.append(run)

    langsmith = type(sys)("langsmith")
    langsmith.Client = _Client
    monkeypatch.setitem(sys.modules, "langsmith", langsmith)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    capped_pack, capped_predictions = _many_held_out(tmp_path / "capped", 5001)
    capped_dest = tmp_path / "capped-score"

    main(
        [
            "score",
            "--pack",
            str(capped_pack),
            "--predictions",
            str(capped_predictions),
            "--dest",
            str(capped_dest),
            "--upload-langsmith",
        ]
    )

    assert len(created) == 1
    assert len(created[0].runs) == 5000
    assert created[0].runs[0]["inputs"]["document_id"] == "doc-0000"
    assert created[0].runs[-1]["inputs"]["document_id"] == "doc-4999"
    capped = json.loads(
        (capped_dest / "scorecard.json").read_text(encoding="utf-8")
    )
    assert capped["traces_sent"] == 5000
    assert capped["candidates"]["tuned"]["support"] == 5001
    assert "test-key" not in (capped_dest / "scorecard.json").read_text(
        encoding="utf-8"
    )


def test_score_rejects_a_prediction_for_the_wrong_contract(tmp_path):
    pack = _held_out_pack(tmp_path)
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            [
                {
                    "document_id": "someone-else",
                    "user": "Quote the Governing Law clause or answer none.",
                    "prediction": "Nevada",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "score",
                "--pack",
                str(pack),
                "--predictions",
                str(predictions),
                "--dest",
                str(tmp_path / "score"),
            ]
        )
    assert "someone-else" in str(exc.value)


def _pack(tmp_path: Path, context: str) -> Path:
    (tmp_path / "labels.csv").write_text(
        "document_id,split,context,Governing Law,Parties\n"
        f"contract-a,train,{context},Nevada,Acme\n",
        encoding="utf-8",
    )
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "id: mini\n"
        "dev_fraction: 0\n"
        "sources:\n"
        "  - id: mini\n"
        "    licence: cc-by-4.0\n"
        f"    path: {tmp_path}\n"
        "attribution:\n"
        '  - "The Atticus Project, CUAD v1, CC BY 4.0"\n',
        encoding="utf-8",
    )
    return pack


def _held_out_pack(tmp_path: Path) -> Path:
    (tmp_path / "labels.csv").write_text(
        "document_id,split,Governing Law\nheld-span,test,Nevada\n",
        encoding="utf-8",
    )
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "id: mini\n"
        "dev_fraction: 0\n"
        "sources:\n"
        "  - id: mini\n"
        "    licence: cc-by-4.0\n"
        f"    path: {tmp_path}\n",
        encoding="utf-8",
    )
    return pack


def _many_held_out(tmp_path: Path, count: int) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    lines = ["document_id,split,Governing Law"]
    rows = []
    question = "Quote the Governing Law clause or answer none."
    for index in range(count):
        document_id = f"doc-{index:04d}"
        lines.append(f"{document_id},test,Nevada")
        rows.append(
            {
                "document_id": document_id,
                "user": question,
                "prediction": "Nevada",
            }
        )
    (tmp_path / "labels.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "id: many\n"
        "dev_fraction: 0\n"
        "sources:\n"
        "  - id: many\n"
        "    licence: cc-by-4.0\n"
        f"    path: {tmp_path}\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps(rows), encoding="utf-8")
    return pack, predictions

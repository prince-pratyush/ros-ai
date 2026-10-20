import json
from pathlib import Path

import pytest

from rosai.cost import CostCeilingError
from rosai.eval_batch import evaluate_batch
from rosai.prepare import InstructionPair, Prepared, SplitProof

# 600s and 1800s on preemptible A100-80GB, 8 cores, 32 GiB.
# Hand sums of the 19 Oct 2026 list prices.
# Lean eval asks for one retry, so the guard prices two attempts.
LEAN_ESTIMATE = 1.043808
FULL_ESTIMATE = 1.565712

# Two of the four answers match the gold span. 2 / 4 = 0.5.
EXACT_MATCH = 0.5
_COMPLETIONS = ("Nevada", "none", "Delaware", "Nevada")
_GOLDS = ("Nevada", "none", "Nevada", "none")


class _FakeEngine:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self._left = list(_COMPLETIONS)

    def generate(self, batch: list[str]) -> list[str]:
        self.batches.append(list(batch))
        out = self._left[: len(batch)]
        self._left = self._left[len(batch) :]
        return out


def _prepared() -> Prepared:
    pairs = tuple(
        InstructionPair(
            document_id=f"doc-{index}",
            task="clause_span",
            messages=(("user", f"prompt-{index}"),),
            completion=gold,
        )
        for index, gold in enumerate(_GOLDS, start=1)
    )
    return Prepared(
        attribution=(),
        split_proof=SplitProof((), (), tuple(pair.document_id for pair in pairs)),
        train=(),
        dev=(),
        held_out=pairs,
    )


def test_execute_eval_writes_one_limited_window_and_strips_the_end_token(
    tmp_path, capsys
):
    from rosai.modal_app import execute_eval

    context = ("A" * 16) + ("B" * 16)
    payload = _held_out_archive("contract-c", context, span=("B" * 16, 16))
    config = (
        Path("configs/modal-eval-lean.yaml")
        .read_text(encoding="utf-8")
        .replace("max_seq_length: 2048", "max_seq_length: 40")
        .replace("limit: 32", "limit: 1")
    )
    engine = _RecordingEngine("guess<|im_end|>\n")

    count = execute_eval(
        config,
        "id: mini\ndev_fraction: 0\n",
        tmp_path / "volume",
        archive=payload,
        engine=engine,
    )

    question = "Quote the Governing Law clause or answer none."
    prompt = (
        "<|im_start|>user\n"
        f"{'A' * 16}\n\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )
    assert count == 1
    assert engine.batches == [[prompt]]
    written = json.loads(
        (tmp_path / "volume" / "predictions.json").read_text(encoding="utf-8")
    )
    assert written == [
        {
            "document_id": "contract-c",
            "user": f"{'A' * 16}\n\n{question}",
            "prediction": "guess",
        }
    ]
    assert "B" * 16 not in prompt
    assert "1.043808" in capsys.readouterr().out


class _RecordingEngine:
    def __init__(self, text: str) -> None:
        self.text = text
        self.batches: list[list[str]] = []

    def generate(self, batch: list[str]) -> list[str]:
        self.batches.append(list(batch))
        return [self.text for _ in batch]


def _held_out_archive(title: str, context: str, *, span: tuple[str, int]) -> bytes:
    import io
    import zipfile

    text, start = span
    item = {
        "title": title,
        "paragraphs": [
            {
                "context": context,
                "qas": [
                    {
                        "id": f"{title}__Governing Law_0",
                        "question": (
                            "Highlight the parts (if any) of this contract related to "
                            '"Governing Law" that should be reviewed by a lawyer.'
                        ),
                        "answers": [{"text": text, "answer_start": start}],
                        "is_impossible": False,
                    }
                ],
            }
        ],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "train_separate_questions.json",
            json.dumps({"version": "1", "data": []}),
        )
        archive.writestr(
            "test.json", json.dumps({"version": "1", "data": [item]})
        )
    return buffer.getvalue()


def test_execute_eval_without_a_limit_keeps_every_held_out_window(tmp_path):
    from rosai.modal_app import execute_eval

    config = (
        Path("configs/modal-eval-lean.yaml")
        .read_text(encoding="utf-8")
        .replace("limit: 32\n", "")
        .replace("max_seq_length: 2048", "max_seq_length: 40")
        .replace("batch_size: 8", "batch_size: 1")
    )
    engine = _RecordingEngine("kept")

    count = execute_eval(
        config,
        "id: mini\ndev_fraction: 0\n",
        tmp_path / "volume",
        archive=_held_out_archive(
            "contract-c", ("A" * 16) + ("B" * 16), span=("B" * 16, 16)
        ),
        engine=engine,
    )

    assert count == 2
    assert [len(batch) for batch in engine.batches] == [1, 1]
    written = json.loads(
        (tmp_path / "volume" / "predictions.json").read_text(encoding="utf-8")
    )
    assert [row["user"][:16] for row in written] == ["A" * 16, "B" * 16]
    assert [row["prediction"] for row in written] == ["kept", "kept"]


def test_execute_eval_refuses_a_ceiling_breach_before_generation(tmp_path):
    from rosai.cost import CostCeilingError
    from rosai.modal_app import execute_eval

    config = (
        Path("configs/modal-eval-lean.yaml")
        .read_text(encoding="utf-8")
        .replace("gpu_seconds: 600", "gpu_seconds: 10800")
    )
    engine = _RecordingEngine("nope")

    with pytest.raises(CostCeilingError):
        execute_eval(
            config,
            "id: mini\ndev_fraction: 0\n",
            tmp_path / "volume",
            archive=b"not-read",
            engine=engine,
        )

    assert engine.batches == []
    assert not (tmp_path / "volume" / "predictions.json").exists()


def test_execute_eval_refuses_a_missing_adapter_without_importing_vllm(tmp_path):
    import sys

    from rosai.modal_app import EvalError, execute_eval

    with pytest.raises(EvalError) as exc:
        execute_eval(
            Path("configs/modal-eval-lean.yaml").read_text(encoding="utf-8"),
            "id: mini\ndev_fraction: 0\n",
            tmp_path / "volume",
            archive=_held_out_archive("contract-c", "Nevada", span=("Nevada", 0)),
        )

    assert "adapter" in str(exc.value)
    assert "vllm" not in sys.modules


def test_vllm_engine_loads_qwen3_with_the_saved_adapter(tmp_path, monkeypatch):
    import sys
    import types

    from rosai.eval_batch import VllmEngine

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 8}\n', encoding="utf-8")
    seen: dict = {}

    class _SamplingParams:
        def __init__(self, *, temperature: float, max_tokens: int) -> None:
            seen["temperature"] = temperature
            seen["max_tokens"] = max_tokens

    class _LLM:
        def __init__(self, **kwargs) -> None:
            seen["llm"] = kwargs

        def generate(self, batch, sampling_params=None, lora_request=None):
            seen["batch"] = list(batch)
            seen["lora_path"] = lora_request.path
            return [_Request("Nevada<|im_end|>")]

    class _Request:
        def __init__(self, text: str) -> None:
            self.outputs = [types.SimpleNamespace(text=text)]

    class _LoRARequest:
        def __init__(self, name: str, lora_int_id: int, path: str) -> None:
            self.name = name
            self.path = path

    vllm = types.ModuleType("vllm")
    vllm.LLM = _LLM
    vllm.SamplingParams = _SamplingParams
    lora_pkg = types.ModuleType("vllm.lora")
    lora_request = types.ModuleType("vllm.lora.request")
    lora_request.LoRARequest = _LoRARequest
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora_pkg)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", lora_request)

    engine = VllmEngine("Qwen/Qwen3-8B", adapter_dir=str(adapter))
    assert engine.generate(["prompt"]) == ["Nevada<|im_end|>"]
    assert seen["llm"] == {
        "model": "Qwen/Qwen3-8B",
        "enable_lora": True,
        "max_lora_rank": 8,
    }
    assert seen["temperature"] == 0
    assert seen["max_tokens"] == 1024
    assert seen["lora_path"] == str(adapter)
    assert seen["batch"] == ["prompt"]


def test_fake_engine_batches_prompts_and_scorecard_matches_exact_count(tmp_path):
    engine = _FakeEngine()
    dest = tmp_path / "run"

    evaluate_batch(_prepared(), engine, dest, batch_size=2)

    assert engine.batches == [
        ["prompt-1", "prompt-2"],
        ["prompt-3", "prompt-4"],
    ]
    card = json.loads((dest / "scorecard.json").read_text(encoding="utf-8"))
    assert card["candidates"]["base"]["exact_match"] == EXACT_MATCH
    assert card["candidates"]["tuned"]["exact_match"] == EXACT_MATCH
    assert card["candidates"]["base"]["support"] == 4


def test_lean_eval_is_capped_and_full_eval_is_not(capsys):
    from rosai.modal_app import function_spec, plan_eval

    lean = plan_eval("configs/modal-eval-lean.yaml")
    assert lean.limit == 32
    assert lean.capped is True
    assert lean.max_usd == 6
    assert lean.retries == 1
    assert lean.estimate_usd == LEAN_ESTIMATE
    full = plan_eval("configs/modal-eval-full.yaml")
    assert full.limit is None
    assert full.capped is False
    assert full.retries == 0
    assert full.estimate_usd == FULL_ESTIMATE
    spec = function_spec(full)
    assert "nonpreemptible" not in spec
    assert spec["timeout"] == 1800
    assert spec["retries"] == 0
    printed = capsys.readouterr().out
    assert "1.043808" in printed
    assert "1.565712" in printed


def test_eleven_retries_on_a_lean_eval_exceed_its_ceiling(tmp_path):
    from rosai.modal_app import plan_eval

    text = Path("configs/modal-eval-lean.yaml").read_text(encoding="utf-8")
    path = tmp_path / "preempt.yaml"
    path.write_text(text.replace("retries: 1", "retries: 11"), encoding="utf-8")
    with pytest.raises(CostCeilingError) as exc:
        plan_eval(path)
    # 12 * 600 * (0.000694 + 8 * 0.0000131 + 32 * 0.00000222)
    assert "6.262848" in str(exc.value)


def test_eval_over_six_dollars_is_refused(tmp_path):
    from rosai.modal_app import plan_eval

    path = tmp_path / "long.yaml"
    text = Path("configs/modal-eval-full.yaml").read_text(encoding="utf-8")
    text = text.replace("gpu_seconds: 1800", "gpu_seconds: 10800")
    text = text.replace("max_gpu_hours: 1", "max_gpu_hours: 4")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(CostCeilingError) as exc:
        plan_eval(path)
    assert "6.00" in str(exc.value)


def test_volume_completions_keep_the_hand_counted_exact_match(tmp_path):
    from rosai.eval_batch import run_eval
    from rosai.modal_app import plan_eval

    lean = plan_eval("configs/modal-eval-lean.yaml")
    engine = _FakeEngine()
    dest = tmp_path / "lean"
    card = run_eval(
        _prepared(),
        engine,
        dest,
        tmp_path / "volume",
        batch_size=2,
        limit=lean.limit,
    )
    assert card.capped is True
    written = json.loads((dest / "scorecard.json").read_text(encoding="utf-8"))
    assert written["capped"] is True
    assert written["candidates"]["tuned"]["exact_match"] == EXACT_MATCH
    full = plan_eval("configs/modal-eval-full.yaml")
    full_dest = tmp_path / "full"
    full_card = run_eval(
        _prepared(),
        _FakeEngine(),
        full_dest,
        tmp_path / "volume-full",
        batch_size=2,
        limit=full.limit,
    )
    assert full_card.capped is False
    assert full_card.candidates["base"].exact_match == EXACT_MATCH


def test_eval_planning_does_not_import_modal_or_vllm():
    import sys

    import rosai.modal_app as app

    app.plan_eval("configs/modal-eval-lean.yaml")
    assert "modal" not in sys.modules
    assert "vllm" not in sys.modules

import json
from pathlib import Path

import pytest

from rosai.cost import CostCeilingError
from rosai.eval_batch import evaluate_batch
from rosai.prepare import InstructionPair, Prepared, SplitProof

# 600s and 1800s on preemptible A100-80GB, 8 cores, 32 GiB.
# Hand sums of the 19 Oct 2026 list prices.
LEAN_ESTIMATE = 0.521904
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
    assert lean.estimate_usd == LEAN_ESTIMATE
    full = plan_eval("configs/modal-eval-full.yaml")
    assert full.limit is None
    assert full.capped is False
    assert full.estimate_usd == FULL_ESTIMATE
    spec = function_spec(full)
    assert "nonpreemptible" not in spec
    assert spec["timeout"] == 1800
    printed = capsys.readouterr().out
    assert "0.521904" in printed
    assert "1.565712" in printed


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

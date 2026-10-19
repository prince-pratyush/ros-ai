import json

from rosai.eval_batch import evaluate_batch
from rosai.prepare import InstructionPair, Prepared, SplitProof

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

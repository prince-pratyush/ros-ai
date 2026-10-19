import json
from pathlib import Path

from rosai.pack import load_pack
from rosai.prepare import InstructionPair, Prepared, SplitProof, prepare
from rosai.score import score

# gold tokens: nevada. prediction tokens: nevada, law.
# overlap 1, precision 1/2, recall 1, F1 = 2 * 0.5 * 1 / 1.5.
NEVADA_LAW_F1 = 0.6666666666666666


class _Scripted:
    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self.text = text

    def complete(self, pair: InstructionPair) -> str:
        return self.text


def _held_out(document_id: str, gold: str) -> Prepared:
    pair = InstructionPair(
        document_id=document_id,
        task="clause_span",
        messages=(("user", "Quote the Governing Law clause or answer none."),),
        completion=gold,
    )
    return Prepared(
        attribution=(),
        split_proof=SplitProof((), (), (document_id,)),
        train=(),
        dev=(),
        held_out=(pair,),
    )


def test_nevada_law_is_not_an_exact_span_and_has_token_f1():
    result = score(_held_out("doc", "Nevada"), [_Scripted("base", "Nevada law")])
    scored = result.candidates["base"]
    assert scored.exact_match == 0
    assert scored.token_f1 == NEVADA_LAW_F1
    assert scored.support == 1
    assert scored.format_failures == 0


def test_none_against_none_is_exact():
    result = score(_held_out("doc", "none"), [_Scripted("base", "none")])
    scored = result.candidates["base"]
    assert scored.exact_match == 1
    assert scored.format_failures == 0


def test_a_paragraph_where_a_span_was_required_is_a_format_failure():
    prediction = "The governing law is stated below.\nNevada law applies."
    result = score(_held_out("doc", "Nevada"), [_Scripted("base", prediction)])
    scored = result.candidates["base"]
    assert scored.format_failures == 1
    assert scored.exact_match == 0


class _Answers:
    def __init__(self, name: str, answers: dict[str, str]) -> None:
        self.name = name
        self.answers = answers
        self.prompts: list[tuple[tuple[str, str], ...]] = []

    def complete(self, pair: InstructionPair) -> str:
        self.prompts.append(pair.messages)
        return self.answers[pair.document_id]


def _score_pack(tmp_path: Path) -> Path:
    (tmp_path / "labels.csv").write_text(
        "document_id,split,Governing Law\n"
        "train-pos,train,Nevada\n"
        "train-empty,train,\n"
        "held-span,test,Nevada\n"
        "held-empty,test,\n",
        encoding="utf-8",
    )
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "id: score-mini\n"
        "dev_fraction: 0\n"
        "negative_ratio: 1\n"
        "sources:\n"
        "  - id: score-mini\n"
        "    licence: cc-by-4.0\n"
        f"    path: {tmp_path}\n"
        "attribution:\n"
        '  - "The Atticus Project, CUAD v1, CC BY 4.0"\n',
        encoding="utf-8",
    )
    return pack


def test_scorecard_keeps_the_held_out_mix_for_base_and_tuned(tmp_path):
    prepared = prepare(load_pack(_score_pack(tmp_path)))
    base = _Answers("base", {"held-span": "Nevada", "held-empty": "none"})
    tuned = _Answers("tuned", {"held-span": "Nevada law", "held-empty": "none"})
    dest = tmp_path / "run"

    score(prepared, [base, tuned], dest)

    assert base.prompts == tuned.prompts
    assert len(base.prompts) == 2
    card = json.loads((dest / "scorecard.json").read_text(encoding="utf-8"))
    assert card["decoding"] == "greedy"
    assert card["capped"] is False
    assert card["candidates"]["base"]["support"] == 2
    assert card["candidates"]["tuned"]["support"] == 2
    assert {row["gold"] for row in card["rows"]} == {"Nevada", "none"}
    assert "hosted" not in card
    assert "judge" not in card
    text = (dest / "scorecard.json").read_text(encoding="utf-8").lower()
    assert "hosted" not in text
    assert "judge" not in text


def test_limit_32_labels_the_scorecard_capped(tmp_path):
    prepared = prepare(load_pack(_score_pack(tmp_path)))
    base = _Answers("base", {"held-span": "Nevada", "held-empty": "none"})
    tuned = _Answers("tuned", {"held-span": "Nevada", "held-empty": "none"})
    dest = tmp_path / "run"

    score(prepared, [base, tuned], dest, limit=32)

    card = json.loads((dest / "scorecard.json").read_text(encoding="utf-8"))
    assert card["capped"] is True
    assert card["candidates"]["base"]["support"] == 2

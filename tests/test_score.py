from rosai.prepare import InstructionPair, Prepared, SplitProof
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

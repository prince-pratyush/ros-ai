import json
import sys

from rosai.prepare import InstructionPair, Prepared, SplitProof
from rosai.score import score

# The key below is a fixture. This test never reads a real credential.
_FAKE_KEY = "test-key"


class _FakeClient:
    def __init__(self) -> None:
        self.runs: list[dict] = []

    def create_run(self, **run) -> None:
        self.runs.append(run)


class _Echo:
    def __init__(self) -> None:
        self.name = "base"

    def complete(self, pair: InstructionPair) -> str:
        return pair.completion


def _prepared(count: int) -> Prepared:
    pairs = tuple(
        InstructionPair(
            document_id=f"doc-{index}",
            task="clause_span",
            messages=(("user", f"prompt-{index}"),),
            completion="Nevada" if index % 2 == 0 else "none",
        )
        for index in range(count)
    )
    return Prepared(
        attribution=(),
        split_proof=SplitProof((), (), tuple(pair.document_id for pair in pairs)),
        train=(),
        dev=(),
        held_out=pairs,
    )


def test_six_thousand_examples_record_five_thousand(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", _FAKE_KEY)
    client = _FakeClient()
    dest = tmp_path / "run"

    score(
        _prepared(6000),
        [_Echo()],
        dest,
        upload=True,
        client=client,
        max_traces=5000,
    )

    assert len(client.runs) == 5000
    assert client.runs[0]["inputs"]["document_id"] == "doc-0"
    assert client.runs[-1]["inputs"]["document_id"] == "doc-4999"
    card = json.loads((dest / "scorecard.json").read_text(encoding="utf-8"))
    assert card["traces_sent"] == 5000
    assert len(card["rows"]) == 6000
    assert card["candidates"]["base"]["support"] == 6000
    assert _FAKE_KEY not in (dest / "scorecard.json").read_text(encoding="utf-8")
    assert "langsmith" not in sys.modules


def test_default_max_traces_is_5000(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", _FAKE_KEY)
    client = _FakeClient()

    score(_prepared(6000), [_Echo()], tmp_path / "run", upload=True, client=client)

    assert len(client.runs) == 5000
    assert "langsmith" not in sys.modules


def test_unset_key_uploads_nothing_and_writes_the_local_file(tmp_path, monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    client = _FakeClient()
    dest = tmp_path / "run"

    score(_prepared(4), [_Echo()], dest, upload=True, client=client)

    assert client.runs == []
    card = json.loads((dest / "scorecard.json").read_text(encoding="utf-8"))
    assert card["traces_sent"] == 0
    assert card["decoding"] == "greedy"
    assert len(card["rows"]) == 4
    assert card["candidates"]["base"]["support"] == 4
    assert card["candidates"]["base"]["exact_match"] == 1
    assert "langsmith" not in sys.modules

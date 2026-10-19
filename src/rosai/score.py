"""Score held-out span answers and write the local scorecard."""

import json
from dataclasses import dataclass
from pathlib import Path

from rosai.metrics import exact_match, format_failure, token_f1
from rosai.prepare import Prepared
from rosai.sinks import DEFAULT_MAX_TRACES, upload_traces


@dataclass(frozen=True)
class CandidateScore:
    exact_match: float
    token_f1: float
    support: int
    format_failures: int


@dataclass(frozen=True)
class Scorecard:
    candidates: dict[str, CandidateScore]
    decoding: str = "greedy"
    capped: bool = False
    traces_sent: int = 0


def score(
    prepared: Prepared,
    candidates,
    dest=None,
    *,
    limit=None,
    upload: bool = False,
    client=None,
    max_traces: int = DEFAULT_MAX_TRACES,
) -> Scorecard:
    examples = prepared.held_out
    if limit is not None:
        examples = examples[: limit]
    rows = [
        {"document_id": pair.document_id, "gold": pair.completion} for pair in examples
    ]
    scored: dict[str, CandidateScore] = {}
    traces: list[dict] = []
    for candidate in candidates:
        exact_total = 0
        f1_total = 0.0
        failures = 0
        for pair in examples:
            prediction = candidate.complete(pair)
            traces.append(
                {
                    "name": candidate.name,
                    "run_type": "chain",
                    "inputs": {"document_id": pair.document_id},
                    "outputs": {"completion": prediction},
                }
            )
            exact_total += exact_match(prediction, pair.completion)
            f1_total += token_f1(prediction, pair.completion)
            if format_failure(prediction, pair.completion):
                failures += 1
        support = len(examples)
        scored[candidate.name] = CandidateScore(
            exact_match=exact_total / support if support else 0.0,
            token_f1=f1_total / support if support else 0.0,
            support=support,
            format_failures=failures,
        )
    sent = upload_traces(traces, client, upload=upload, max_traces=max_traces)
    card = Scorecard(
        candidates=scored,
        decoding="greedy",
        capped=limit is not None,
        traces_sent=sent,
    )
    if dest is not None:
        _write(Path(dest), card, rows)
    return card


def _write(dest: Path, card: Scorecard, rows: list[dict[str, str]]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "decoding": card.decoding,
        "capped": card.capped,
        "traces_sent": card.traces_sent,
        "candidates": {
            name: {
                "exact_match": item.exact_match,
                "token_f1": item.token_f1,
                "support": item.support,
                "format_failures": item.format_failures,
            }
            for name, item in card.candidates.items()
        },
        "rows": rows,
    }
    (dest / "scorecard.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

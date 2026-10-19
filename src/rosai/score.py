"""Score held-out span answers from scripted or live candidates."""

from dataclasses import dataclass

from rosai.metrics import exact_match, format_failure, token_f1
from rosai.prepare import Prepared


@dataclass(frozen=True)
class CandidateScore:
    exact_match: float
    token_f1: float
    support: int
    format_failures: int


@dataclass(frozen=True)
class Scorecard:
    candidates: dict[str, CandidateScore]


def score(prepared: Prepared, candidates) -> Scorecard:
    scored: dict[str, CandidateScore] = {}
    for candidate in candidates:
        exact_total = 0
        f1_total = 0.0
        failures = 0
        for pair in prepared.held_out:
            prediction = candidate.complete(pair)
            exact_total += exact_match(prediction, pair.completion)
            f1_total += token_f1(prediction, pair.completion)
            if format_failure(prediction, pair.completion):
                failures += 1
        support = len(prepared.held_out)
        scored[candidate.name] = CandidateScore(
            exact_match=exact_total / support if support else 0.0,
            token_f1=f1_total / support if support else 0.0,
            support=support,
            format_failures=failures,
        )
    return Scorecard(candidates=scored)

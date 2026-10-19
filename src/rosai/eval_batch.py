"""Run held-out prompts through an engine in batches and score them."""

from rosai.prepare import Prepared
from rosai.score import score


class _Replay:
    def __init__(self, name: str, completions: list[str]) -> None:
        self.name = name
        self._left = list(completions)

    def complete(self, pair) -> str:
        return self._left.pop(0)


def run_batches(engine, prompts: list[str], batch_size: int) -> list[str]:
    completions: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start : start + batch_size])
        completions.extend(engine.generate(batch))
    return completions


def evaluate_batch(prepared: Prepared, engine, dest, *, batch_size: int, limit=None):
    examples = prepared.held_out if limit is None else prepared.held_out[:limit]
    prompts = [pair.messages[-1][1] for pair in examples]
    completions = run_batches(engine, prompts, batch_size)
    return score(
        prepared,
        [_Replay("base", completions), _Replay("tuned", completions)],
        dest,
        limit=limit,
    )

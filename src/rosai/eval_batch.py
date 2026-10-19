"""Run held-out prompts through an engine in batches and score them."""

import json
from pathlib import Path

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
    completions = _complete(prepared, engine, batch_size, limit)
    return _score(prepared, completions, dest, limit)


def run_eval(prepared: Prepared, engine, dest, volume_dir, *, batch_size: int, limit):
    completions = _complete(prepared, engine, batch_size, limit)
    write_completions(volume_dir, completions)
    return _score(prepared, read_completions(volume_dir), dest, limit)


def write_completions(directory, completions: list[str]) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / "completions.json").write_text(
        json.dumps(completions) + "\n",
        encoding="utf-8",
    )


def read_completions(directory) -> list[str]:
    return json.loads(
        (Path(directory) / "completions.json").read_text(encoding="utf-8")
    )


class VllmEngine:
    """vLLM adapter. The import happens on the first batch, not at startup."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._llm = None

    def generate(self, batch: list[str]) -> list[str]:
        if self._llm is None:
            from vllm import LLM

            self._llm = LLM(model=self.model_id)
        outputs = self._llm.generate(batch)
        return [output.outputs[0].text for output in outputs]


def _complete(prepared: Prepared, engine, batch_size: int, limit) -> list[str]:
    examples = prepared.held_out if limit is None else prepared.held_out[:limit]
    prompts = [pair.messages[-1][1] for pair in examples]
    return run_batches(engine, prompts, batch_size)


def _score(prepared: Prepared, completions: list[str], dest, limit):
    return score(
        prepared,
        [_Replay("base", completions), _Replay("tuned", completions)],
        dest,
        limit=limit,
    )

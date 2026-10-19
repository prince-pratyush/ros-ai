"""Exact match and token F1 for a span answer, plus format failures."""

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = set(string.punctuation)


def exact_match(prediction: str, gold: str) -> int:
    return int(_normalize(prediction) == _normalize(gold))


def token_f1(prediction: str, gold: str) -> float:
    predicted = _tokens(prediction)
    expected = _tokens(gold)
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def format_failure(prediction: str, gold: str) -> bool:
    """A paragraph is a format failure when the gold answer is a span."""
    if _normalize(gold) == "none":
        return False
    return "\n" in prediction and "\n" not in gold


def _normalize(text: str) -> str:
    lowered = text.lower()
    stripped = "".join(ch for ch in lowered if ch not in _PUNCTUATION)
    without_articles = _ARTICLES.sub(" ", stripped)
    return " ".join(without_articles.split())


def _tokens(text: str) -> list[str]:
    normalized = _normalize(text)
    if not normalized:
        return []
    return normalized.split()

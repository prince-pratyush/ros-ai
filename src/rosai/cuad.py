"""Read clause spans from a pinned CUAD data.zip."""

import json
import re
import zipfile
from io import BytesIO

_CATEGORY = re.compile(r'related to "([^"]+)"')


class CategoryError(Exception):
    """A pack named a clause category the archive does not contain."""


def clause_rows(
    payload: bytes, categories: tuple[str, ...]
) -> tuple[tuple[str, str, str, str], ...]:
    """Return (document_id, split, category, span). An empty span means none."""
    archive = zipfile.ZipFile(BytesIO(payload))
    found: set[str] = set()
    rows: list[tuple[str, str, str, str]] = []
    for name, split in (
        ("train_separate_questions.json", "train"),
        ("test.json", "test"),
    ):
        data = json.loads(archive.read(name))
        for contract in data["data"]:
            grouped = {category: [] for category in categories}
            for paragraph in contract["paragraphs"]:
                for qa in paragraph["qas"]:
                    match = _CATEGORY.search(qa["question"])
                    if match is None:
                        continue
                    category = match.group(1)
                    found.add(category)
                    if category not in grouped:
                        continue
                    for answer in qa["answers"]:
                        text = str(answer.get("text") or "").strip()
                        if text and text not in grouped[category]:
                            grouped[category].append(text)
            document_id = str(contract["title"]).strip()
            for category in categories:
                rows.append(
                    (document_id, split, category, " | ".join(grouped[category]))
                )
    missing = [category for category in categories if category not in found]
    if missing:
        raise CategoryError(f"{missing[0]} is not a CUAD category")
    return tuple(rows)

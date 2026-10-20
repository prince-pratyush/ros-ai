"""Read clause spans from a pinned CUAD data.zip."""

import json
import re
import zipfile
from io import BytesIO
from pathlib import Path

import yaml

_CATEGORY = re.compile(r'related to "([^"]+)"')

# Names as they appear in the pinned data.zip questions, in inventory order.
CUAD_CATEGORIES: tuple[str, ...] = (
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "Competitive Restriction Exception",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "Ip Ownership Assignment",
    "Joint Ip Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
)


class CategoryError(Exception):
    """A pack named a clause category the archive does not contain."""


def load_categories(path: str | Path) -> tuple[str, ...]:
    """Clause categories a run config names. Absent means Governing Law."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    listed = raw.get("categories") if isinstance(raw, dict) else None
    if not listed:
        return ("Governing Law",)
    names = tuple(str(item) for item in listed)
    known = set(CUAD_CATEGORIES)
    for name in names:
        if name not in known:
            raise CategoryError(f"{name} is not a CUAD category")
    return names


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

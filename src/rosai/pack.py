"""Load a domain pack and refuse a source whose licence is off the allowlist."""

from dataclasses import dataclass
from pathlib import Path

import yaml

ALLOWED_LICENCES = frozenset({"cc-by-4.0", "cc0-1.0", "apache-2.0"})


class LicenceError(Exception):
    """A pack named a source whose licence is not allowed."""


@dataclass(frozen=True)
class Source:
    id: str
    licence: str


@dataclass(frozen=True)
class Pack:
    id: str
    sources: tuple[Source, ...]
    attribution: tuple[str, ...]


def load_pack(path: str | Path) -> Pack:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise LicenceError(f"{path}: pack file is empty")
    sources = tuple(_source(item) for item in raw.get("sources") or [])
    attribution = tuple(raw.get("attribution") or ())
    return Pack(id=str(raw.get("id") or ""), sources=sources, attribution=attribution)


def _source(item: object) -> Source:
    if not isinstance(item, dict):
        raise LicenceError("source entry is missing an id and a licence")
    source_id = str(item.get("id") or "")
    licence = item.get("licence")
    if not isinstance(licence, str):
        licence = ""
    if licence not in ALLOWED_LICENCES:
        raise LicenceError(f"{source_id}: licence {licence!r} is not allowed")
    return Source(id=source_id, licence=licence)

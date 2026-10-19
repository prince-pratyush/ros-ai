"""Turn a local CUAD-shaped pack into clause-span pairs and a dev split."""

import csv
from dataclasses import dataclass
from pathlib import Path

from rosai.pack import Pack

_TASK = "clause_span"
_CATEGORY = "Governing Law"


@dataclass(frozen=True)
class InstructionPair:
    document_id: str
    task: str
    messages: tuple[tuple[str, str], ...]
    completion: str


@dataclass(frozen=True)
class SplitProof:
    """train_ids is the train-side pool. dev_ids is carved from that pool."""

    train_ids: tuple[str, ...]
    dev_ids: tuple[str, ...]
    held_out_ids: tuple[str, ...]


@dataclass(frozen=True)
class Prepared:
    attribution: tuple[str, ...]
    split_proof: SplitProof
    train: tuple[InstructionPair, ...]
    dev: tuple[InstructionPair, ...]
    held_out: tuple[InstructionPair, ...]


def prepare(pack: Pack) -> Prepared:
    rows = _rows(pack)
    train_pool = tuple(sorted(row[0] for row in rows if row[1] == "train"))
    held_out_ids = tuple(sorted(row[0] for row in rows if row[1] == "test"))
    dev_count = int(len(train_pool) * pack.dev_fraction)
    dev_ids = train_pool[-dev_count:] if dev_count else ()
    dev_set = set(dev_ids)
    by_id = {row[0]: row for row in rows}
    train_pairs = tuple(
        _pair(by_id[document_id])
        for document_id in train_pool
        if document_id not in dev_set
    )
    dev_pairs = tuple(_pair(by_id[document_id]) for document_id in dev_ids)
    held_out_pairs = tuple(
        _pair(by_id[document_id]) for document_id in held_out_ids
    )
    return Prepared(
        attribution=pack.attribution,
        split_proof=SplitProof(
            train_ids=train_pool,
            dev_ids=dev_ids,
            held_out_ids=held_out_ids,
        ),
        train=train_pairs,
        dev=dev_pairs,
        held_out=held_out_pairs,
    )


def _rows(pack: Pack) -> tuple[tuple[str, str, str], ...]:
    found: list[tuple[str, str, str]] = []
    for source in pack.sources:
        if not source.path:
            continue
        path = Path(source.path) / "labels.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                found.append(
                    (
                        record["document_id"].strip(),
                        record["split"].strip(),
                        record[_CATEGORY].strip(),
                    )
                )
    return tuple(found)


def _pair(row: tuple[str, str, str]) -> InstructionPair:
    document_id, _split, span = row
    completion = span if span else "none"
    question = "Quote the Governing Law clause or answer none."
    return InstructionPair(
        document_id=document_id,
        task=_TASK,
        messages=(("user", question),),
        completion=completion,
    )

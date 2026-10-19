from pathlib import Path

import pytest

from rosai.pack import load_pack
from rosai.prepare import SplitError, prepare


def test_prepare_emits_span_or_none_and_carves_dev_from_train():
    prepared = prepare(load_pack(Path("tests/fixtures/cuad_mini/pack.yaml")))

    by_document = {
        pair.document_id: pair.completion
        for pair in (*prepared.train, *prepared.dev)
    }
    assert by_document == {"contract-a": "Nevada", "contract-b": "none"}

    proof = prepared.split_proof
    assert set(proof.dev_ids) <= set(proof.train_ids)
    assert proof.dev_ids
    train_ids = {pair.document_id for pair in prepared.train}
    dev_ids = {pair.document_id for pair in prepared.dev}
    assert dev_ids.isdisjoint(train_ids)
    assert dev_ids == set(proof.dev_ids)

    assert any("CC BY 4.0" in line for line in prepared.attribution)


def test_prepare_fails_when_a_document_id_is_in_two_splits():
    with pytest.raises(SplitError) as exc:
        prepare(load_pack(Path("tests/fixtures/cuad_mini/overlap/pack.yaml")))
    assert "contract-a" in str(exc.value)

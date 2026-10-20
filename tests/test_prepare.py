from pathlib import Path

import pytest

from rosai.cuad import CategoryError
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


def test_labels_quote_each_requested_category(tmp_path):
    (tmp_path / "labels.csv").write_text(
        "document_id,split,Governing Law,Parties\n"
        "contract-a,train,Nevada,Acme\n"
        "contract-b,train,,Beta\n",
        encoding="utf-8",
    )
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "id: mini\n"
        "dev_fraction: 0.5\n"
        "sources:\n"
        "  - id: mini\n"
        "    licence: cc-by-4.0\n"
        f"    path: {tmp_path}\n"
        "attribution:\n"
        '  - "The Atticus Project, CUAD v1, CC BY 4.0"\n',
        encoding="utf-8",
    )

    prepared = prepare(
        load_pack(pack), categories=("Governing Law", "Parties")
    )

    assert [
        (pair.document_id, pair.messages[0][1], pair.completion)
        for pair in prepared.train
    ] == [
        ("contract-a", "Quote the Governing Law clause or answer none.", "Nevada"),
        ("contract-a", "Quote the Parties clause or answer none.", "Acme"),
    ]
    assert [
        (pair.document_id, pair.completion) for pair in prepared.dev
    ] == [
        ("contract-b", "none"),
        ("contract-b", "Beta"),
    ]


def test_labels_reject_a_missing_category_column(tmp_path):
    (tmp_path / "labels.csv").write_text(
        "document_id,split,Governing Law\ncontract-a,train,Nevada\n",
        encoding="utf-8",
    )
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "id: mini\n"
        "sources:\n"
        "  - id: mini\n"
        "    licence: cc-by-4.0\n"
        f"    path: {tmp_path}\n",
        encoding="utf-8",
    )

    with pytest.raises(CategoryError) as exc:
        prepare(load_pack(pack), categories=("Parties",))
    assert "Parties" in str(exc.value)


def test_prepare_fails_when_a_document_id_is_in_two_splits():
    with pytest.raises(SplitError) as exc:
        prepare(load_pack(Path("tests/fixtures/cuad_mini/overlap/pack.yaml")))
    assert "contract-a" in str(exc.value)

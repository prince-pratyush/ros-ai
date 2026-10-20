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


def test_a_label_file_quotes_the_contract_window_that_holds_the_span(tmp_path):
    context = ("A" * 16) + "Nevada" + ("B" * 10)
    (tmp_path / "labels.csv").write_text(
        "document_id,split,context,Governing Law\n"
        f"contract-a,train,{context},Nevada\n",
        encoding="utf-8",
    )
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "id: mini\n"
        "dev_fraction: 0\n"
        "sources:\n"
        "  - id: mini\n"
        "    licence: cc-by-4.0\n"
        f"    path: {tmp_path}\n",
        encoding="utf-8",
    )
    question = "Quote the Governing Law clause or answer none."

    prepared = prepare(load_pack(pack), max_seq_length=40)

    assert [
        (pair.messages[0][1], pair.completion) for pair in prepared.train
    ] == [
        (f"{'A' * 16}\n\n{question}", "none"),
        (f"Nevada{'B' * 10}\n\n{question}", "Nevada"),
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


def test_a_long_contract_is_split_into_windows_that_keep_their_spans(tmp_path):
    # max_seq_length 40 budgets 16 characters of contract per window:
    # half of the 8 tokens left after the chat wrapper, at 4 characters each.
    context = ("A" * 16) + ("B" * 16) + ("C" * 16)
    question = "Quote the Governing Law clause or answer none."
    payload = _archive_context(
        (
            (
                "contract-a",
                "train",
                context,
                {"Governing Law": ((("B" * 16), 16),)},
            ),
        )
    )
    pack = _pinned(tmp_path, dev_fraction="0")

    prepared = prepare(
        load_pack(pack), archive=payload, max_seq_length=40
    )

    assert [
        (pair.messages[0][1], pair.completion) for pair in prepared.train
    ] == [
        (f"{'A' * 16}\n\n{question}", "none"),
        (f"{'B' * 16}\n\n{question}", "B" * 16),
        (f"{'C' * 16}\n\n{question}", "none"),
    ]
    assert prepared.dev == ()
    assert prepared.held_out == ()
    assert prepared.split_proof.train_ids == ("contract-a",)


def test_windows_stay_on_their_contract_split(tmp_path):
    context = ("A" * 16) + ("B" * 16)
    payload = _archive_context(
        (
            ("contract-a", "train", context, {"Governing Law": ()}),
            (
                "contract-b",
                "train",
                context,
                {"Governing Law": ((("B" * 16), 16),)},
            ),
            (
                "contract-c",
                "test",
                context,
                {"Governing Law": ((("A" * 16), 0),)},
            ),
        )
    )
    pack = _pinned(tmp_path, dev_fraction="0.5")

    prepared = prepare(load_pack(pack), archive=payload, max_seq_length=40)

    assert [pair.document_id for pair in prepared.train] == [
        "contract-a",
        "contract-a",
    ]
    assert [pair.completion for pair in prepared.train] == ["none", "none"]
    assert [pair.document_id for pair in prepared.dev] == [
        "contract-b",
        "contract-b",
    ]
    assert [pair.completion for pair in prepared.dev] == ["none", "B" * 16]
    assert [pair.document_id for pair in prepared.held_out] == [
        "contract-c",
        "contract-c",
    ]
    assert [pair.completion for pair in prepared.held_out] == ["A" * 16, "none"]
    assert prepared.split_proof.train_ids == ("contract-a", "contract-b")
    assert prepared.split_proof.dev_ids == ("contract-b",)
    assert prepared.split_proof.held_out_ids == ("contract-c",)
    assert {pair.document_id for pair in prepared.train}.isdisjoint(
        prepared.split_proof.held_out_ids
    )


def test_a_span_in_a_later_paragraph_uses_the_joined_contract(tmp_path):
    question = "Quote the Governing Law clause or answer none."
    payload = _archive_paragraphs(
        "contract-a",
        ("A" * 16, "B" * 16),
        spans=((("B" * 16), 0),),
    )
    pack = _pinned(tmp_path, dev_fraction="0")

    prepared = prepare(load_pack(pack), archive=payload, max_seq_length=40)

    assert [
        (pair.messages[0][1], pair.completion) for pair in prepared.train
    ] == [
        (f"{'A' * 16}\n\n{question}", "none"),
        (f"\n{'B' * 15}\n\n{question}", "B" * 16),
        (f"B\n\n{question}", "none"),
    ]


def _pinned(tmp_path: Path, *, dev_fraction: str) -> Path:
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "id: tiny\n"
        f"dev_fraction: {dev_fraction}\n"
        "sources:\n"
        "  - id: tiny\n"
        "    licence: cc-by-4.0\n"
        "    url: https://example.test/data.zip\n"
        "    sha256: abc\n"
        "attribution:\n"
        '  - "The Atticus Project, CUAD v1, CC BY 4.0"\n',
        encoding="utf-8",
    )
    return pack


def _archive_paragraphs(title: str, paragraphs: tuple[str, ...], *, spans) -> bytes:
    import io
    import json
    import zipfile

    qas = [
        {
            "id": f"{title}__Governing Law_0",
            "question": (
                "Highlight the parts (if any) of this contract "
                'related to "Governing Law" that should be reviewed by a lawyer.'
            ),
            "answers": [
                {"text": text, "answer_start": start} for text, start in spans
            ],
            "is_impossible": not spans,
        }
    ]
    item = {
        "title": title,
        "paragraphs": [
            {"context": paragraphs[0], "qas": []},
            {"context": paragraphs[1], "qas": qas},
        ],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "train_separate_questions.json",
            json.dumps({"version": "1", "data": [item]}),
        )
        archive.writestr("test.json", json.dumps({"version": "1", "data": []}))
    return buffer.getvalue()


def _archive_context(contracts) -> bytes:
    import io
    import json
    import zipfile

    train: list[dict] = []
    test: list[dict] = []
    for title, split, context, spans_by_category in contracts:
        qas = []
        for category, spans in spans_by_category.items():
            qas.append(
                {
                    "id": f"{title}__{category}_0",
                    "question": (
                        "Highlight the parts (if any) of this contract related to "
                        f'"{category}" that should be reviewed by a lawyer.'
                    ),
                    "answers": [
                        {"text": text, "answer_start": start}
                        for text, start in spans
                    ],
                    "is_impossible": not spans,
                }
            )
        item = {
            "title": title,
            "paragraphs": [{"context": context, "qas": qas}],
        }
        (test if split == "test" else train).append(item)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "train_separate_questions.json",
            json.dumps({"version": "1", "data": train}),
        )
        archive.writestr("test.json", json.dumps({"version": "1", "data": test}))
    return buffer.getvalue()

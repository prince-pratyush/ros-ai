import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from rosai.cuad import CategoryError
from rosai.fetch import OfflineError, cached_download
from rosai.pack import Source, load_pack
from rosai.prepare import SplitError, prepare

_PUBLISHED_TRAIN = "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT"
_PUBLISHED_TEST = (
    "LohaCompanyltd_20191209_F-1_EX-10.16_11917878_EX-10.16_Supply Agreement"
)

_GOVERNING_LAW = "Quote the Governing Law clause or answer none."


def test_cache_reuses_matching_bytes_and_replaces_a_bad_file(tmp_path):
    payload = b"cuad-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    source = Source(
        id="theatticusproject/cuad",
        licence="cc-by-4.0",
        url="https://example.test/data.zip",
        sha256=digest,
    )
    calls = {"n": 0}

    def read(url: str) -> bytes:
        calls["n"] += 1
        assert url == source.url
        return payload

    assert cached_download(source, tmp_path, read=read) == payload
    assert cached_download(source, tmp_path, read=read) == payload
    assert calls["n"] == 1
    (tmp_path / digest).write_bytes(b"corrupt")
    assert cached_download(source, tmp_path, read=read) == payload
    assert calls["n"] == 2
    assert (tmp_path / digest).read_bytes() == payload


def test_prepare_quotes_the_requested_category(tmp_path):
    payload = _archive(
        (("contract-a", "train", ("Acme Corp",)),),
        category="Parties",
    )
    pack = _pinned_pack(tmp_path, dev_fraction="0")

    prepared = prepare(load_pack(pack), archive=payload, categories=("Parties",))

    assert [(pair.document_id, pair.completion) for pair in prepared.train] == [
        ("contract-a", "Acme Corp")
    ]
    assert prepared.train[0].messages == (
        ("user", "Quote the Parties clause or answer none."),
    )
    assert prepared.dev == ()
    assert prepared.held_out == ()


def test_each_category_stays_with_its_document(tmp_path):
    payload = _archive_categories(
        (
            (
                "contract-a",
                "train",
                {"Governing Law": ("Nevada",), "Parties": ("Acme",)},
            ),
            (
                "contract-b",
                "train",
                {"Governing Law": (), "Parties": ("Beta",)},
            ),
            (
                "contract-c",
                "test",
                {"Governing Law": ("New York",), "Parties": ("Gamma",)},
            ),
        )
    )
    pack = _pinned_pack(tmp_path, dev_fraction="0.5")

    prepared = prepare(
        load_pack(pack),
        archive=payload,
        categories=("Governing Law", "Parties"),
    )

    assert [
        (pair.document_id, pair.messages[0][1], pair.completion)
        for pair in prepared.train
    ] == [
        ("contract-a", "Quote the Governing Law clause or answer none.", "Nevada"),
        ("contract-a", "Quote the Parties clause or answer none.", "Acme"),
    ]
    assert [
        (pair.document_id, pair.messages[0][1], pair.completion)
        for pair in prepared.dev
    ] == [
        ("contract-b", "Quote the Governing Law clause or answer none.", "none"),
        ("contract-b", "Quote the Parties clause or answer none.", "Beta"),
    ]
    assert [
        (pair.document_id, pair.messages[0][1], pair.completion)
        for pair in prepared.held_out
    ] == [
        ("contract-c", "Quote the Governing Law clause or answer none.", "New York"),
        ("contract-c", "Quote the Parties clause or answer none.", "Gamma"),
    ]
    proof = prepared.split_proof
    assert proof.train_ids == ("contract-a", "contract-b")
    assert proof.dev_ids == ("contract-b",)
    assert proof.held_out_ids == ("contract-c",)
    assert {pair.document_id for pair in prepared.train}.isdisjoint(proof.dev_ids)


def test_cached_archive_honours_the_published_split(tmp_path):
    payload = _archive_categories(
        (
            (
                _PUBLISHED_TRAIN,
                "train",
                {"Governing Law": ("Nevada",), "Parties": ("Acme",)},
            ),
            (
                _PUBLISHED_TEST,
                "test",
                {"Governing Law": ("New York",), "Parties": ("Gamma",)},
            ),
        )
    )
    pack = _published_pack(tmp_path)

    prepared = prepare(
        load_pack(pack),
        archive=payload,
        categories=("Governing Law", "Parties"),
    )

    assert prepared.split_proof.train_ids == (_PUBLISHED_TRAIN,)
    assert prepared.split_proof.held_out_ids == (_PUBLISHED_TEST,)
    assert [pair.document_id for pair in prepared.train] == [
        _PUBLISHED_TRAIN,
        _PUBLISHED_TRAIN,
    ]
    assert [pair.completion for pair in prepared.train] == ["Nevada", "Acme"]
    assert [pair.document_id for pair in prepared.held_out] == [
        _PUBLISHED_TEST,
        _PUBLISHED_TEST,
    ]
    assert [pair.completion for pair in prepared.held_out] == ["New York", "Gamma"]
    assert prepared.dev == ()


def test_published_archive_rejects_a_test_contract_marked_train(tmp_path):
    payload = _archive(((_PUBLISHED_TEST, "train", ("Nevada",)),))
    pack = _published_pack(tmp_path)

    with pytest.raises(SplitError) as exc:
        prepare(load_pack(pack), archive=payload)
    assert _PUBLISHED_TEST in str(exc.value)


def test_prepare_rejects_an_unknown_category(tmp_path):
    payload = _archive(
        (("contract-a", "train", ("somewhere",)),),
        category="Not A Clause",
    )
    pack = _pinned_pack(tmp_path, dev_fraction="0")

    with pytest.raises(CategoryError) as exc:
        prepare(
            load_pack(pack),
            archive=payload,
            categories=("Not A Clause",),
        )
    assert "Not A Clause" in str(exc.value)


def test_archive_pairs_quote_governing_law_spans(tmp_path):
    payload = _archive(
        (
            ("contract-a", "train", ("Nevada", "Delaware")),
            ("contract-b", "train", ()),
            ("contract-c", "test", ("New York",)),
        )
    )
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "id: tiny\n"
        "dev_fraction: 0.5\n"
        "sources:\n"
        "  - id: tiny\n"
        "    licence: cc-by-4.0\n"
        "    url: https://example.test/data.zip\n"
        "    sha256: abc\n"
        "attribution:\n"
        '  - "The Atticus Project, CUAD v1, CC BY 4.0"\n',
        encoding="utf-8",
    )

    prepared = prepare(load_pack(pack), archive=payload)

    assert [(pair.document_id, pair.completion) for pair in prepared.train] == [
        ("contract-a", "Nevada | Delaware")
    ]
    assert [(pair.document_id, pair.completion) for pair in prepared.dev] == [
        ("contract-b", "none")
    ]
    assert [(pair.document_id, pair.completion) for pair in prepared.held_out] == [
        ("contract-c", "New York")
    ]
    assert prepared.train[0].messages == (("user", _GOVERNING_LAW),)
    assert any("CC BY 4.0" in line for line in prepared.attribution)


def test_prepare_reads_a_cached_archive_offline(tmp_path, monkeypatch):
    payload = _archive((("contract-a", "train", ("Nevada",)),))
    digest = hashlib.sha256(payload).hexdigest()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / digest).write_bytes(payload)
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "id: tiny\n"
        "dev_fraction: 0\n"
        "sources:\n"
        "  - id: tiny\n"
        "    licence: cc-by-4.0\n"
        "    url: https://example.test/data.zip\n"
        f"    sha256: {digest}\n"
        "attribution:\n"
        '  - "The Atticus Project, CUAD v1, CC BY 4.0"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("ROSAI_DOWNLOAD_CUAD", raising=False)

    def explode(url):
        raise AssertionError(url)

    monkeypatch.setattr("urllib.request.urlopen", explode)

    prepared = prepare(load_pack(pack), cache_dir=cache)

    assert [(pair.document_id, pair.completion) for pair in prepared.train] == [
        ("contract-a", "Nevada")
    ]
    assert prepared.held_out == ()


def test_legal_pack_stays_offline_without_the_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("ROSAI_DOWNLOAD_CUAD", raising=False)

    def explode(url):
        raise AssertionError(url)

    monkeypatch.setattr("urllib.request.urlopen", explode)
    with pytest.raises(OfflineError) as exc:
        prepare(load_pack(Path("packs/legal.yaml")), cache_dir=tmp_path)
    assert "ROSAI_DOWNLOAD_CUAD" in str(exc.value)


def _published_pack(tmp_path: Path) -> Path:
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "id: tiny\n"
        "split: published\n"
        "dev_fraction: 0\n"
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


def _pinned_pack(tmp_path: Path, *, dev_fraction: str) -> Path:
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


def _archive_categories(
    contracts: tuple[tuple[str, str, dict[str, tuple[str, ...]]], ...],
) -> bytes:
    train: list[dict] = []
    test: list[dict] = []
    for title, split, spans_by_category in contracts:
        qas = []
        for category, spans in spans_by_category.items():
            qas.append(
                {
                    "id": f"{title}__{category}_0",
                    "question": (
                        "Highlight the parts (if any) of this contract related to "
                        f'"{category}" that should be reviewed by a lawyer.'
                    ),
                    "answers": [{"text": text, "answer_start": 0} for text in spans],
                    "is_impossible": not spans,
                }
            )
        item = {"title": title, "paragraphs": [{"context": "x", "qas": qas}]}
        (test if split == "test" else train).append(item)
    return _zip_cuad(train, test)


def _archive(
    contracts: tuple[tuple[str, str, tuple[str, ...]], ...],
    *,
    category: str = "Governing Law",
) -> bytes:
    train: list[dict] = []
    test: list[dict] = []
    for title, split, spans in contracts:
        qas = [
            {
                "id": f"{title}__{category}_0",
                "question": (
                    "Highlight the parts (if any) of this contract related to "
                    f'"{category}" that should be reviewed by a lawyer.'
                ),
                "answers": [{"text": text, "answer_start": 0} for text in spans],
                "is_impossible": not spans,
            }
        ]
        item = {"title": title, "paragraphs": [{"context": "x", "qas": qas}]}
        (test if split == "test" else train).append(item)
    return _zip_cuad(train, test)


def _zip_cuad(train: list[dict], test: list[dict]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "train_separate_questions.json",
            json.dumps({"version": "1", "data": train}),
        )
        archive.writestr("test.json", json.dumps({"version": "1", "data": test}))
    return buffer.getvalue()

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from rosai.fetch import OfflineError, cached_download
from rosai.pack import Source, load_pack
from rosai.prepare import prepare

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


def test_legal_pack_stays_offline_without_the_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("ROSAI_DOWNLOAD_CUAD", raising=False)

    def explode(url):
        raise AssertionError(url)

    monkeypatch.setattr("urllib.request.urlopen", explode)
    with pytest.raises(OfflineError) as exc:
        prepare(load_pack(Path("packs/legal.yaml")), cache_dir=tmp_path)
    assert "ROSAI_DOWNLOAD_CUAD" in str(exc.value)


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
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "train_separate_questions.json",
            json.dumps({"version": "1", "data": train}),
        )
        archive.writestr("test.json", json.dumps({"version": "1", "data": test}))
    return buffer.getvalue()

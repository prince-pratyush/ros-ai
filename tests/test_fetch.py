import pytest

from rosai.fetch import ChecksumError, download
from rosai.pack import Source

# sha256 of the bytes b"not-a-zip", computed outside this test.
NOT_A_ZIP_SHA256 = "f300f6a921e42e9c177aa31cb9085db9067972ed4901536757e8e339158c6814"


def test_download_rejects_a_mismatched_checksum():
    source = Source(
        id="theatticusproject/cuad",
        licence="cc-by-4.0",
        url="https://example.invalid/data.zip",
        sha256="0" * 64,
    )
    with pytest.raises(ChecksumError) as exc:
        download(source, read=lambda url: b"not-a-zip")
    assert "theatticusproject/cuad" in str(exc.value)


def test_download_returns_bytes_when_the_checksum_matches():
    source = Source(
        id="theatticusproject/cuad",
        licence="cc-by-4.0",
        url="https://example.invalid/data.zip",
        sha256=NOT_A_ZIP_SHA256,
    )
    seen: list[str] = []

    def read(url: str) -> bytes:
        seen.append(url)
        return b"not-a-zip"

    assert download(source, read=read) == b"not-a-zip"
    assert seen == ["https://example.invalid/data.zip"]

"""Download a pinned source and refuse bytes that miss its sha256."""

import hashlib
import urllib.request
from collections.abc import Callable
from pathlib import Path

from rosai.pack import Source


class ChecksumError(Exception):
    """Downloaded bytes did not match the pinned sha256."""


class OfflineError(Exception):
    """A pinned download was requested without ROSAI_DOWNLOAD_CUAD=1."""


def download(source: Source, read: Callable[[str], bytes] | None = None) -> bytes:
    if not source.url or not source.sha256:
        raise ChecksumError(f"{source.id}: missing a pinned url or sha256")
    payload = (read or _read)(source.url)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != source.sha256:
        raise ChecksumError(
            f"{source.id}: sha256 {actual} does not match {source.sha256}"
        )
    return payload


def cached_download(
    source: Source,
    cache_dir: str | Path,
    read: Callable[[str], bytes] | None = None,
) -> bytes:
    path = Path(cache_dir) / source.sha256
    if path.is_file():
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() == source.sha256:
            return payload
    payload = download(source, read=read)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _read(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()

"""Download a pinned source and refuse bytes that miss its sha256."""

import hashlib
import urllib.request
from collections.abc import Callable

from rosai.pack import Source


class ChecksumError(Exception):
    """Downloaded bytes did not match the pinned sha256."""


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


def _read(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()

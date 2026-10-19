from pathlib import Path

import pytest

from rosai.pack import LicenceError, load_pack


def test_package_imports():
    import rosai

    assert rosai.__doc__


def test_nc_sa_pack_is_refused():
    with pytest.raises(LicenceError) as exc:
        load_pack(Path("tests/fixtures/nc_sa_pack.yaml"))
    message = str(exc.value)
    assert "pile-of-law/pile-of-law" in message
    assert "cc-by-nc-sa-4.0" in message


def test_empty_licence_is_refused(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text(
        "id: refused\nsources:\n  - id: example/empty\n    licence: ''\n",
        encoding="utf-8",
    )
    with pytest.raises(LicenceError) as exc:
        load_pack(path)
    message = str(exc.value)
    assert "example/empty" in message
    assert "''" in message


def test_cc0_and_apache_licences_pass(tmp_path):
    for licence in ("cc0-1.0", "apache-2.0"):
        path = tmp_path / f"{licence}.yaml"
        path.write_text(
            f"id: ok\nsources:\n  - id: example/{licence}\n    licence: {licence}\n",
            encoding="utf-8",
        )
        pack = load_pack(path)
        assert pack.sources[0].licence == licence


def test_legal_pack_records_cc_by_and_atticus_attribution():
    pack = load_pack(Path("packs/legal.yaml"))
    assert {source.id for source in pack.sources} == {
        "theatticusproject/cuad",
        "theatticusproject/cuad-qa",
    }
    assert {source.licence for source in pack.sources} == {"cc-by-4.0"}
    assert any("The Atticus Project" in line for line in pack.attribution)


def test_cc_by_nc_pack_is_refused(tmp_path):
    path = tmp_path / "nc.yaml"
    path.write_text(
        "id: refused\nsources:\n"
        "  - id: example/consumer-contracts\n"
        "    licence: cc-by-nc-4.0\n",
        encoding="utf-8",
    )
    with pytest.raises(LicenceError) as exc:
        load_pack(path)
    message = str(exc.value)
    assert "example/consumer-contracts" in message
    assert "cc-by-nc-4.0" in message

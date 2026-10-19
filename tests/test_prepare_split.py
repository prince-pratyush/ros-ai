import os
from pathlib import Path

import pytest

from rosai.pack import load_pack
from rosai.prepare import SplitError, prepare

# Titles copied from TheAtticusProject/cuad data.zip, in file order.
# Train: the first three contracts in train_separate_questions.json.
# Test: the first three contracts in test.json.
PUBLISHED_TRAIN = (
    "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT",
    "WHITESMOKE,INC_11_08_2011-EX-10.26-PROMOTION AND DISTRIBUTION AGREEMENT",
    "NELNETINC_04_08_2020-EX-1-JOINT FILING AGREEMENT",
)
PUBLISHED_TEST = (
    "LohaCompanyltd_20191209_F-1_EX-10.16_11917878_EX-10.16_Supply Agreement",
    "CENTRACKINTERNATIONALINC_10_29_1999-EX-10.3-WEB SITE HOSTING AGREEMENT",
    "DovaPharmaceuticalsInc_20181108_10-Q_EX-10.2_11414857_EX-10.2_Promotion Agreement",
)
PINNED_REVISION = "6d9a118fb629814c4384a1e63cfc84f5c01894b3"


def test_legal_pack_pins_cuad_qa_revision():
    pack = load_pack(Path("packs/legal.yaml"))
    assert pack.split == "published"
    by_id = {source.id: source for source in pack.sources}
    assert by_id["theatticusproject/cuad-qa"].revision == PINNED_REVISION


def test_published_train_and_test_ids_stay_disjoint():
    prepared = prepare(load_pack(Path("tests/fixtures/cuad_published/pack.yaml")))
    proof = prepared.split_proof
    assert set(proof.held_out_ids) == set(PUBLISHED_TEST)
    assert set(proof.train_ids) == set(PUBLISHED_TRAIN)
    assert set(proof.dev_ids) <= set(PUBLISHED_TRAIN)
    assert proof.dev_ids
    assert set(proof.dev_ids).isdisjoint(proof.held_out_ids)
    train_ids = {pair.document_id for pair in prepared.train}
    dev_ids = {pair.document_id for pair in prepared.dev}
    held_out_ids = {pair.document_id for pair in prepared.held_out}
    assert train_ids.isdisjoint(dev_ids)
    assert train_ids.isdisjoint(held_out_ids)
    assert dev_ids.isdisjoint(held_out_ids)
    assert train_ids | dev_ids == set(PUBLISHED_TRAIN)


def test_published_test_id_labelled_as_train_fails():
    with pytest.raises(SplitError) as exc:
        prepare(load_pack(Path("tests/fixtures/cuad_published_mislabelled/pack.yaml")))
    assert PUBLISHED_TEST[0] in str(exc.value)


@pytest.mark.skipif(
    os.environ.get("ROSAI_DOWNLOAD_CUAD") != "1",
    reason="set ROSAI_DOWNLOAD_CUAD=1 to download CUAD",
)
def test_checked_in_ids_match_published_cuad_files():
    import json
    import urllib.request
    import zipfile
    from io import BytesIO

    commits = json.load(
        urllib.request.urlopen(
            "https://huggingface.co/api/datasets/theatticusproject/cuad-qa/commits/main"
        )
    )
    assert commits[0]["id"] == PINNED_REVISION
    payload = urllib.request.urlopen(
        "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
    ).read()
    archive = zipfile.ZipFile(BytesIO(payload))

    def titles(name: str) -> set[str]:
        data = json.loads(archive.read(name))
        return {example.get("title", "").strip() for example in data["data"]}

    train_titles = titles("train_separate_questions.json")
    test_titles = titles("test.json")
    assert set(PUBLISHED_TRAIN) <= train_titles
    assert set(PUBLISHED_TEST) <= test_titles
    assert set(PUBLISHED_TRAIN).isdisjoint(test_titles)
    assert set(PUBLISHED_TEST).isdisjoint(train_titles)

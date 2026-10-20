import tomllib
from pathlib import Path

import pytest

from rosai.cost import CostCeilingError

# Two attempts (retries: 1) of 1200s preemptible A100-80GB, 8 cores, 32 GiB.
# Hand sum of the 19 Oct 2026 rates.
LEAN_ESTIMATE = 2.087616
# 10800s, same machine.
FULL_ESTIMATE = 9.394272
_BANNED = ("token", "secret", "password", "api_key", "sk-")


def test_lean_config_is_capped_and_under_three_dollars(capsys):
    from rosai.modal_app import plan_train

    plan = plan_train("configs/modal-lean.yaml")
    assert plan.limit == 256
    assert plan.max_usd == 3
    assert plan.epochs == 1
    assert plan.retries == 1
    assert plan.estimate_usd == LEAN_ESTIMATE
    assert plan.gpu == "A100-80GB"
    assert plan.volume_name == "rosai-checkpoints"
    assert "2.087616" in capsys.readouterr().out


def test_full_config_is_one_uncapped_epoch_under_twelve_dollars(capsys):
    from rosai.modal_app import function_spec, plan_train

    plan = plan_train("configs/modal-full.yaml")
    assert plan.limit is None
    assert plan.epochs == 1
    assert plan.max_usd == 12
    assert plan.max_gpu_hours == 3
    assert plan.retries == 0
    assert plan.estimate_usd == FULL_ESTIMATE
    assert plan.checkpoint_every == 50
    spec = function_spec(plan)
    assert "nonpreemptible" not in spec
    assert spec["gpu"] == "A100-80GB"
    assert spec["cpu"] == 8
    assert spec["memory"] == 32768
    assert spec["timeout"] == 10800
    assert spec["retries"] == 0
    assert spec["volumes"] == {"/checkpoints": "rosai-checkpoints"}
    assert "9.394272" in capsys.readouterr().out


def test_resume_starts_at_the_recorded_checkpoint_step(tmp_path):
    from rosai.modal_app import plan_train, record_checkpoint

    assert plan_train("configs/modal-full.yaml", checkpoint_dir=tmp_path).start_step == 0
    record_checkpoint(tmp_path, 120)
    resumed = plan_train("configs/modal-full.yaml", checkpoint_dir=tmp_path)
    assert resumed.start_step == 120


def test_a_lean_train_without_retries_still_prices_one_retry(tmp_path):
    from rosai.modal_app import plan_train

    text = Path("configs/modal-lean.yaml").read_text(encoding="utf-8")
    text = text.replace("retries: 1\n", "")
    path = tmp_path / "default.yaml"
    path.write_text(text, encoding="utf-8")
    plan = plan_train(path)
    assert plan.retries == 1
    assert plan.estimate_usd == LEAN_ESTIMATE


def test_three_retries_on_a_lean_train_exceed_its_ceiling(tmp_path):
    from rosai.modal_app import plan_train

    text = Path("configs/modal-lean.yaml").read_text(encoding="utf-8")
    path = tmp_path / "preempt.yaml"
    path.write_text(text.replace("retries: 1", "retries: 3"), encoding="utf-8")
    with pytest.raises(CostCeilingError) as exc:
        plan_train(path)
    assert "4.175232" in str(exc.value)


def test_full_shape_on_h100_is_refused(tmp_path):
    from rosai.modal_app import plan_train

    path = tmp_path / "h100.yaml"
    path.write_text(
        Path("configs/modal-full.yaml")
        .read_text(encoding="utf-8")
        .replace("A100-80GB", "H100"),
        encoding="utf-8",
    )
    with pytest.raises(CostCeilingError):
        plan_train(path)


def test_modal_configs_contain_no_secrets():
    for name in ("configs/modal-lean.yaml", "configs/modal-full.yaml"):
        text = Path(name).read_text(encoding="utf-8").lower()
        for banned in _BANNED:
            assert banned not in text


def test_modal_extra_is_not_part_of_dev_and_planning_skips_the_sdk():
    import sys

    import rosai.modal_app as app

    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    assert extras["modal"] == ["modal"]
    assert "modal" not in extras["dev"]
    app.plan_train("configs/modal-lean.yaml")
    assert "modal" not in sys.modules

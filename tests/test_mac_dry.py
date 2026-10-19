import json
import os
import subprocess
import sys
from pathlib import Path

from rosai.train import load_run

# sha256 of configs/mac-dry.yaml, computed outside this test.
CONFIG_HASH = "176ec0a6cadb5a6f6efe503856ef15869b2c8887797a60cd5fa65870fa779f4f"


def test_mac_dry_config_is_the_small_fake_run():
    run = load_run(Path("configs/mac-dry.yaml"))
    assert run.model_id == "Qwen/Qwen3-0.6B"
    assert run.thinking is False
    assert run.max_seq_length == 1024
    assert run.limit == 32
    assert run.backend == "fake"
    assert run.batch_size == 1
    assert run.loss == "completion"
    assert run.seed == 3407
    assert run.config_hash == CONFIG_HASH


def test_cpu_smoke_writes_a_manifest_without_gpu_imports(tmp_path):
    result = _run_train(
        pack="tests/fixtures/cuad_mini/pack.yaml",
        config="configs/mac-dry.yaml",
        dest=tmp_path,
        block_network=True,
    )
    assert result.returncode == 0, result.stderr
    assert _forbidden(result.stderr) == ""
    written = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert written["model_id"] == "Qwen/Qwen3-0.6B"
    assert written["thinking"] is False
    assert written["max_seq_length"] == 1024
    assert written["backend"] == "fake"
    assert written["limit"] == 32
    assert written["example_count"] == 1


def test_pair_cap_limits_what_the_fake_backend_fits(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "document_id,split,Governing Law\n"
        "contract-a,train,Nevada\n"
        "contract-b,train,\n"
        "contract-c,train,Delaware\n",
        encoding="utf-8",
    )
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "id: cap\n"
        "dev_fraction: 0\n"
        "sources:\n"
        "  - id: cap\n"
        "    licence: cc-by-4.0\n"
        f"    path: {tmp_path}\n"
        "attribution:\n"
        '  - "The Atticus Project, CUAD v1, CC BY 4.0"\n',
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        Path("configs/mac-dry.yaml")
        .read_text(encoding="utf-8")
        .replace("limit: 32", "limit: 2"),
        encoding="utf-8",
    )
    dest = tmp_path / "run"
    result = _run_train(
        pack=str(pack),
        config=str(config),
        dest=dest,
        block_network=True,
    )
    assert result.returncode == 0, result.stderr
    written = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert written["limit"] == 2
    assert written["example_count"] == 2


def test_mlx_backend_is_rejected_by_the_runner(tmp_path):
    config = tmp_path / "mlx.yaml"
    config.write_text(
        Path("configs/mac-dry.yaml")
        .read_text(encoding="utf-8")
        .replace("backend: fake", "backend: mlx"),
        encoding="utf-8",
    )
    dest = tmp_path / "run"
    result = _run_train(
        pack="tests/fixtures/cuad_mini/pack.yaml",
        config=str(config),
        dest=dest,
        block_network=True,
    )
    assert result.returncode != 0
    assert "invoked by hand" in result.stderr
    assert _forbidden(result.stderr) == ""
    assert not (dest / "manifest.json").exists()


def test_mac_extra_is_not_part_of_the_dev_extra():
    import tomllib

    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    assert extras["mac"] == ["mlx", "mlx-lm"]
    assert extras["dev"] == ["pytest"]


def _forbidden(stderr: str) -> str:
    for line in stderr.splitlines():
        if line.startswith("FORBIDDEN"):
            return line.removeprefix("FORBIDDEN").strip()
    raise AssertionError(stderr)


def _run_train(*, pack: str, config: str, dest: Path, block_network: bool) -> subprocess.CompletedProcess[str]:
    guard = ""
    if block_network:
        guard = """
import socket
import urllib.request
def _blocked(*args, **kwargs):
    raise RuntimeError("network is off")
socket.create_connection = _blocked
urllib.request.urlopen = _blocked
"""
    script = f"""
{guard}
import sys
sys.argv = [
    "rosai", "train",
    "--pack", {pack!r},
    "--config", {config!r},
    "--dest", {str(dest)!r},
]
from rosai.__main__ import main
try:
    main()
finally:
    names = sorted(
        name for name in sys.modules
        if name == "mlx" or name.startswith("mlx.")
        or name == "mlx_lm" or name.startswith("mlx_lm.")
        or name == "unsloth" or name.startswith("unsloth.")
        or name == "modal" or name.startswith("modal.")
    )
    sys.stderr.write("FORBIDDEN " + ",".join(names) + "\\n")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

"""CPU entry points. GPU and MLX backends stay out of this process."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rosai.pack import load_pack
from rosai.prepare import prepare
from rosai.train import FakeBackend, load_run, train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rosai")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("--pack", required=True)

    train_command = commands.add_parser("train")
    train_command.add_argument("--pack", required=True)
    train_command.add_argument("--config", required=True)
    train_command.add_argument("--dest", default="runs/mac-dry")
    train_command.add_argument(
        "--hand",
        action="store_true",
        help="run the MLX fit by hand; the test suite never passes this flag",
    )

    score_command = commands.add_parser("score")
    score_command.add_argument("--pack", required=True)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        _prepare(args.pack)
    elif args.command == "train":
        _train(args.pack, args.config, args.dest, hand=args.hand)
    else:
        raise SystemExit("score is not in this slice")


def _prepare(pack_path: str) -> None:
    prepared = prepare(load_pack(pack_path))
    print(
        f"train {len(prepared.train)} dev {len(prepared.dev)} "
        f"held_out {len(prepared.held_out)}"
    )


def _train(pack_path: str, config_path: str, dest: str, *, hand: bool) -> None:
    run = load_run(config_path)
    if run.backend == "mlx":
        if not hand:
            raise SystemExit(
                "backend 'mlx' must be invoked by hand: "
                "python -m rosai train --pack PACK --config CONFIG --hand"
            )
        _hand_mlx(pack_path, run, Path(dest))
        return
    if run.backend != "fake":
        raise SystemExit(
            f"backend {run.backend!r} is not run by this command"
        )
    if hand:
        raise SystemExit("--hand is only for backend mlx")
    prepared = prepare(load_pack(pack_path))
    manifest = train(prepared, run, FakeBackend(), dest)
    print(
        f"trained {manifest.example_count} examples -> {dest}/manifest.json"
    )


def _hand_mlx(pack_path: str, run, dest: Path) -> None:
    prepared = prepare(load_pack(pack_path))
    examples = prepared.train
    if run.limit is not None:
        examples = examples[: run.limit]
    if not examples:
        raise SystemExit("no training pairs to hand to MLX")
    data = dest / "mlx-data"
    data.mkdir(parents=True, exist_ok=True)
    _write_jsonl(data / "train.jsonl", examples)
    _write_jsonl(data / "valid.jsonl", prepared.dev)
    command = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        run.model_id,
        "--train",
        "--data",
        str(data),
        "--batch-size",
        str(run.batch_size),
        "--max-seq-length",
        str(run.max_seq_length),
        "--iters",
        str(len(examples)),
        "--mask-prompt",
        "--seed",
        str(run.seed),
        "--adapter-path",
        str(dest / "adapters"),
        "--grad-checkpoint",
    ]
    print(" ".join(command), flush=True)
    raise SystemExit(subprocess.call(command))


def _write_jsonl(path: Path, pairs) -> None:
    lines = []
    for pair in pairs:
        messages = [{"role": role, "content": text} for role, text in pair.messages]
        messages.append({"role": "assistant", "content": pair.completion})
        lines.append(json.dumps({"messages": messages}, ensure_ascii=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


if __name__ == "__main__":
    main()

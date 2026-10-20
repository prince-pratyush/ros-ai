"""CPU entry points. GPU and MLX backends stay out of this process."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from rosai.cuad import load_categories
from rosai.pack import load_pack
from rosai.prepare import prepare
from rosai.score import RecordedAnswers, ScoreError, score
from rosai.train import FakeBackend, load_run, train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rosai")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("--pack", required=True)
    prepare_command.add_argument("--config")
    prepare_command.add_argument("--categories", nargs="*")

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
    score_command.add_argument("--predictions", required=True)
    score_command.add_argument("--dest", default="runs/score")
    score_command.add_argument("--config")
    score_command.add_argument("--upload-langsmith", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "prepare":
        _prepare(args.pack, args.config, args.categories)
    elif args.command == "train":
        _train(args.pack, args.config, args.dest, hand=args.hand)
    else:
        _score(
            args.pack,
            args.predictions,
            args.dest,
            args.config,
            upload=args.upload_langsmith,
        )


def _prepare(pack_path: str, config_path: str | None, categories) -> None:
    chosen, max_seq_length, _limit = _budget(config_path, categories)
    prepared = prepare(
        load_pack(pack_path),
        categories=chosen,
        max_seq_length=max_seq_length,
    )
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
        _hand_mlx(pack_path, config_path, run, Path(dest))
        return
    if run.backend != "fake":
        raise SystemExit(
            f"backend {run.backend!r} is not run by this command"
        )
    if hand:
        raise SystemExit("--hand is only for backend mlx")
    prepared = _prepared(pack_path, config_path, run)
    manifest = train(prepared, run, FakeBackend(), dest)
    print(
        f"trained {manifest.example_count} examples -> {dest}/manifest.json"
    )


def _hand_mlx(pack_path: str, config_path: str, run, dest: Path) -> None:
    prepared = _prepared(pack_path, config_path, run)
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


def _score(pack_path, predictions, dest, config_path, *, upload: bool) -> None:
    chosen, max_seq_length, limit = _budget(config_path, None)
    prepared = prepare(
        load_pack(pack_path),
        categories=chosen,
        max_seq_length=max_seq_length,
    )
    rows = json.loads(Path(predictions).read_text(encoding="utf-8"))
    examples = prepared.held_out
    if limit is not None:
        examples = examples[:limit]
    if len(rows) != len(examples):
        raise SystemExit(
            f"predictions cover {len(rows)} pairs "
            f"but the held-out split has {len(examples)}"
        )
    try:
        card = score(
            prepared,
            [RecordedAnswers(rows)],
            dest,
            limit=limit,
            upload=upload,
        )
    except ScoreError as exc:
        raise SystemExit(str(exc)) from exc
    tuned = card.candidates["tuned"]
    print(f"exact_match {tuned.exact_match} token_f1 {tuned.token_f1}")


def _prepared(pack_path: str, config_path: str, run):
    return prepare(
        load_pack(pack_path),
        categories=load_categories(config_path),
        max_seq_length=run.max_seq_length,
    )


def _budget(config_path: str | None, categories):
    chosen = None
    max_seq_length = 2048
    limit = None
    if config_path:
        chosen = load_categories(config_path)
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        model = raw.get("model") or {}
        max_seq_length = int(model.get("max_seq_length", 2048))
        if raw.get("limit") is not None:
            limit = int(raw["limit"])
    if categories:
        chosen = tuple(categories)
    return chosen, max_seq_length, limit


def _write_jsonl(path: Path, pairs) -> None:
    lines = []
    for pair in pairs:
        messages = [{"role": role, "content": text} for role, text in pair.messages]
        messages.append({"role": "assistant", "content": pair.completion})
        lines.append(json.dumps({"messages": messages}, ensure_ascii=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


if __name__ == "__main__":
    main()

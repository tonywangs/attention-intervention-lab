import argparse
import sys
import zipfile
from .artifacts import Cancelled


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline CPU attention intervention lab")
    sub = parser.add_subparsers(dest="command", required=True)
    pilot = sub.add_parser("pilot", help="development only, hard 30 minute budget")
    pilot.add_argument("--output", required=True)
    pilot.add_argument("--steps", type=int, default=1500)
    train = sub.add_parser("train", help="train three seeds from a frozen protocol")
    train.add_argument("--protocol", required=True)
    train.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate", help="evaluate selected checkpoints offline")
    evaluate.add_argument("--training", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--max-seconds", type=float, default=600)
    replay = sub.add_parser("replay", help="replay saved inputs and compare checkpoint predictions")
    replay.add_argument("--training", required=True)
    replay.add_argument("--reference", required=True)
    replay.add_argument("--output", required=True)
    replay.add_argument("--max-seconds", type=float, default=600)
    report = sub.add_parser("report", help="regenerate HTML from verified evaluation artifacts")
    report.add_argument("--input", required=True)
    report.add_argument("--output", required=True)
    for name in ('pair-prepare', 'pairs', 'pair-replay'):
        pair = sub.add_parser(name, help='frozen exhaustive pair experiment')
        pair.add_argument('--training', required=True)
        pair.add_argument('--previous', required=True)
        pair.add_argument('--protocol', required=True)
        pair.add_argument('--output', required=True)
        if name != 'pair-prepare':
            pair.add_argument('--data', required=True)
            pair.add_argument('--max-seconds', type=float, default=7200)
        if name == 'pair-replay':
            pair.add_argument('--reference', required=True)
    pair_report = sub.add_parser('pair-report', help='regenerate the paired intervention report')
    pair_report.add_argument('--input', required=True)
    pair_report.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command in ('pair-prepare', 'pairs', 'pair-replay'):
            from .interactions import prepare, evaluate_pairs, replay_pairs
            if args.command == 'pair-prepare':
                prepare(args.protocol, args.training, args.previous, args.output)
            elif args.command == 'pairs':
                evaluate_pairs(args.training, args.previous, args.protocol, args.data, args.output, args.max_seconds)
            else:
                replay_pairs(args.training, args.previous, args.protocol, args.data, args.reference, args.output, args.max_seconds)
        elif args.command == 'pair-report':
            from .artifacts import transaction, verify_manifest, write_json, manifest
            from .interaction_report import render_directory
            verify_manifest(args.input)
            with transaction(args.output) as out:
                render_directory(args.input, out/'report.html')
                write_json(out/'manifest.json', manifest(out))
        elif args.command == "pilot":
            from .train import pilot
            pilot(args.output, args.steps)
        elif args.command == "train":
            from .train import train_final
            train_final(args.protocol, args.output)
        elif args.command == "evaluate":
            from .evaluate import evaluate
            evaluate(args.training, args.output, args.max_seconds)
        elif args.command == "replay":
            from .evaluate import replay
            replay(args.training, args.reference, args.output, args.max_seconds)
        elif args.command == "report":
            from .artifacts import transaction, verify_manifest, write_json, manifest
            from .report import render_directory
            verify_manifest(args.input)
            with transaction(args.output) as out:
                render_directory(args.input, out / "report.html")
                write_json(out / "manifest.json", manifest(out))
    except (ValueError, OSError, KeyError, TypeError, TimeoutError, Cancelled,
            zipfile.BadZipFile, EOFError) as exc:
        print(f"attention-lab: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

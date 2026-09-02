from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .config import load_config
from .dataset import generate_sources
from .pipeline import run_study


def _snapshot_config(config_path: Path, output: Path) -> Path:
    destination = output / "benchmark_config.yaml"
    if config_path.resolve() == destination.resolve():
        return destination
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(config_path, temporary)
    temporary.replace(destination)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocr-bench",
        description="Controlled 16-pixel English and Traditional Chinese OCR benchmark",
    )
    parser.add_argument(
        "command",
        choices=("generate", "study", "report", "export-comparison", "compare-runs", "all"),
    )
    parser.add_argument("runs", nargs="*", type=Path, help="run directories for compare-runs")
    parser.add_argument("--config", type=Path, help="benchmark YAML configuration")
    parser.add_argument("--output", type=Path, help="run output directory")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace outputs owned by the selected stage",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="write reproducibility manifests, stable PNGs, and exact OCR metadata without changing legacy benchmark semantics",
    )
    parser.add_argument(
        "--deterministic-renderer",
        action="store_true",
        help="with --exact, replace Pillow text rendering with the deterministic in-repo renderer",
    )
    parser.add_argument(
        "--baseline-run",
        type=Path,
        help="with --exact, reuse matching frozen resized pixels from this run",
    )
    parser.add_argument(
        "--natural-input",
        type=Path,
        help="natural-photo input directory for export-comparison",
    )
    parser.add_argument(
        "--imatest-input",
        type=Path,
        help="Imatest-pattern input directory for export-comparison",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "compare-runs":
            if len(args.runs) != 2:
                raise ValueError("compare-runs requires exactly two run directories")
            from .compare_runs import compare_runs

            problems = compare_runs(args.runs[0].resolve(), args.runs[1].resolve())
            if problems:
                for problem in problems:
                    print(problem, file=sys.stderr)
                return 1
            print(f"Runs match: {args.runs[0]} and {args.runs[1]}")
            return 0
        if args.config is None or args.output is None:
            raise ValueError(f"{args.command} requires --config and --output")
        if args.deterministic_renderer and not args.exact:
            raise ValueError("--deterministic-renderer requires --exact")
        if args.baseline_run is not None and not args.exact:
            raise ValueError("--baseline-run requires --exact")
        config = load_config(args.config)
        output = args.output.resolve()
        forbidden_outputs = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
        if output in forbidden_outputs:
            raise ValueError(
                "--output must be a dedicated run directory, not the filesystem root, "
                "home directory, or project root"
            )
        output.mkdir(parents=True, exist_ok=True)
        _snapshot_config(config.source_path, output)
        if args.command in {"generate", "all"}:
            manifest = generate_sources(
                config,
                output,
                force=args.force,
                exact=args.exact,
                deterministic_renderer=args.deterministic_renderer,
            )
            print(f"Generated {len(manifest['sources'])} canonical images in {output}")
        if args.command in {"study", "all"}:
            manifest = run_study(
                config,
                output,
                force=args.force,
                exact=args.exact,
                baseline_run=args.baseline_run,
            )
            print(f"Evaluated {manifest['result_count']} resized images")
        if args.command in {"report", "all"}:
            from .report import generate_report

            plots = generate_report(config, output, force=args.force)
            print(f"Created {len(plots)} report artifacts for {output}")
        if args.command == "export-comparison":
            if args.natural_input is None or args.imatest_input is None:
                raise ValueError(
                    "export-comparison requires --natural-input and --imatest-input"
                )
            from .comparison import export_comparison

            manifest = export_comparison(
                config,
                output,
                args.natural_input,
                args.imatest_input,
                force=args.force,
            )
            count = sum(item["file_count"] for item in manifest["inputs"].values())
            print(f"Exported {count} source images into {output / 'comparison_exports'}")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

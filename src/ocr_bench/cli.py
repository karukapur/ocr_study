from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .dataset import generate_sources
from .pipeline import run_study


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocr-bench",
        description="Controlled 16-pixel English and Traditional Chinese OCR benchmark",
    )
    parser.add_argument(
        "command", choices=("generate", "study", "report", "export-comparison", "all")
    )
    parser.add_argument("--config", required=True, type=Path, help="benchmark YAML configuration")
    parser.add_argument("--output", required=True, type=Path, help="run output directory")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace outputs owned by the selected stage",
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
        config = load_config(args.config)
        output = args.output.resolve()
        forbidden_outputs = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
        if output in forbidden_outputs:
            raise ValueError(
                "--output must be a dedicated run directory, not the filesystem root, "
                "home directory, or project root"
            )
        output.mkdir(parents=True, exist_ok=True)
        if args.command in {"generate", "all"}:
            manifest = generate_sources(config, output, force=args.force)
            print(f"Generated {len(manifest['sources'])} canonical images in {output}")
        if args.command in {"study", "all"}:
            manifest = run_study(config, output, force=args.force)
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

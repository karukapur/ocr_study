"""Compare quantized OpenCV and project floating-point bilinear outputs."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from ocr_bench.resample import floating_point_bilinear  # noqa: E402

from experiments.replicate_bilinear.uint8._compare_uint8 import (  # noqa: E402
    parse_args,
    run_comparison,
)


PATTERN_DIR = Path(__file__).resolve().parent.parent / "test_patterns"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"
DEFAULT_CSV_NAME = "bilinear_floating_uint8_psnr_metrics.csv"


def main() -> None:
    args = parse_args(__doc__ or "", PATTERN_DIR, DEFAULT_OUTPUT)
    run_comparison(
        pattern_dir=args.pattern_dir,
        output_dir=args.output_dir,
        write_images=args.write_images,
        reports_only=args.reports_only,
        candidate_resize=lambda image, ratio_w, ratio_h: floating_point_bilinear(
            image, ratio_w, ratio_h, interpolation="shift"
        ),
        candidate_name="floating_point_bilinear",
        csv_name=DEFAULT_CSV_NAME,
        report_title="OpenCV vs Floating-Point Bilinear Quantized uint8 PSNR",
        ratios=args.ratios,
    )


if __name__ == "__main__":
    main()


# Controlled 16-pixel OCR resampling benchmark

This project compares two integer box-binning modes, OpenCV bilinear resize,
and custom fixed-point Lanczos-2/Lanczos-3 polyphase filters on synthetic
English and Traditional Chinese OCR inputs. Every method uses the same
canonical source image for a pattern and requested ratio.

Each source uses the smallest automatically measured square canvas that fits
the rendered text plus 16 output-domain pixels of padding per side. The square
side is rounded up to a multiple of six so integer binning never drops pixels.

## Prerequisites

- Python 3.10+
- Tesseract 5, with `eng` and `chi_tra` files available in `tessdata/`
- Licensed Consolas and Microsoft JhengHei files placed as described in
  [`fonts/README.md`](fonts/README.md)

On Homebrew systems, Traditional Chinese data is normally provided by
`tesseract-lang`. The program checks prerequisites before doing expensive work
and never substitutes another font.

## Install and run

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m ocr_bench all --config benchmark.yaml --output artifacts/run-001
```

The stages can also be run independently:

```bash
.venv/bin/python -m ocr_bench generate --config benchmark.yaml --output artifacts/run-001
.venv/bin/python -m ocr_bench study --config benchmark.yaml --output artifacts/run-001
.venv/bin/python -m ocr_bench report --config benchmark.yaml --output artifacts/run-001
```

Add `--force` to rebuild the files owned by a stage. Each run records source,
font, tessdata, configuration, and output hashes in JSON manifests.

## Outputs

- `canonical/`: 493 source PNGs (29 patterns × 17 ratios)
- `resized/`: 2,465 method outputs
- `ocr/`: Tesseract text and diagnostics
- `image_results.csv`: one record per evaluated image
- `aggregate_results.csv`: English, Traditional Chinese, and combined CER
- `performance_summary.txt`: concise best/worst cases, methods, and ratios
- `performance_extremes.csv`: structured best/worst summary records
- `plots/`: combined and per-language CER curves, heatmaps, and glyph-height plots
- `montage/`: per-pattern 2.625x visual method comparisons
- `run_manifest.json`: complete reproducibility metadata, including Lanczos
  coefficient banks

At fractional requested ratios, floor and ceiling binning deliberately use an
integer effective factor. For example, 1.625x uses 1x floor binning and 2x
ceiling binning, so those outputs are not expected to retain a 16-pixel glyph.

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

## Natural and Imatest comparison export

After the study stage, natural photographs and Imatest-generated patterns can
be exported through the best and worst non-binning method/ratio conditions:

```bash
.venv/bin/python -m ocr_bench export-comparison \
  --config benchmark.yaml \
  --output artifacts/run-001 \
  --natural-input /path/to/photos \
  --imatest-input /path/to/imatest
```

The command accepts single-frame 8-bit PNG, JPEG, TIFF, and BMP images. It
creates matched candidate/reference PNG trees in `comparison_exports/` for
external Butteraugli, SSIM, MSE, MAE, PSNR, or similar analysis. Binning is
excluded before ranking, and tied conditions are resolved by the lowest ratio
and then the method identifier.

The reference is a separable float64 Lanczos-3 filter with 129 coefficient
slots and 1024 phases. Lanczos-3 is the conventional neutral choice here:
[ImageMagick defines standard Lanczos with three lobes and uses it when
shrinking ordinary photographs](https://usage.imagemagick.org/filter/),
[FFmpeg defaults its Lanczos width/alpha to 3](https://www.ffmpeg.org/ffmpeg-scaler.html),
and [libvips exposes Lanczos-3 as its higher-order Lanczos
kernel](https://libvips.github.io/pyvips/enums.html). ImageMagick also notes
that four lobes can help some fine shrinking patterns at the cost of more
ringing, so Lanczos-4 is not an unqualified higher-quality reference.

## Outputs

- `canonical/`: 493 source PNGs (29 patterns × 17 ratios)
- `resized/`: 2,465 method outputs
- `ocr/`: Tesseract text and diagnostics
- `image_results.csv`: one record per evaluated image
- `aggregate_results.csv`: English, Traditional Chinese, and combined CER
- `performance_summary.txt`: concise best/worst cases, methods, and ratios
- `performance_extremes.csv`: structured best/worst summary records
- `comparison_exports/`: optional natural/Imatest golden and candidate pairs
- `plots/`: combined and per-language CER curves, heatmaps, and glyph-height plots
- `montage/`: per-pattern 2.625x visual method comparisons
- `run_manifest.json`: complete reproducibility metadata, including Lanczos
  coefficient banks

At fractional requested ratios, floor and ceiling binning deliberately use an
integer effective factor. For example, 1.625x uses 1x floor binning and 2x
ceiling binning, so those outputs are not expected to retain a 16-pixel glyph.

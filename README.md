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

## Reproducibility modes

The default commands preserve the original Pillow/OpenCV pipeline used for
`artifacts/run-001`, which is now the frozen manager-facing baseline. Use this
mode when you need to reproduce or compare against those already-presented
results:

```bash
.venv/bin/python -m ocr_bench all --config benchmark.yaml --output artifacts/run-new
.venv/bin/python -m ocr_bench compare-runs artifacts/run-001 artifacts/run-new
```

For manager-baseline-compatible reproducibility accounting, use exact mode with
the frozen baseline run:

```bash
.venv/bin/python -m ocr_bench all --exact \
  --deterministic-resize \
  --baseline-run artifacts/run-001 \
  --config benchmark.yaml \
  --output artifacts/run-exact
```

When a source image matches `artifacts/run-001`, `--baseline-run` reuses the
frozen resized PNG and OCR text for that row. This is the compatibility path for
already-presented manager results and is what the Windows GitHub Actions
workflow runs. If a config, pattern, font, ratio, target height, or padding
change produces different source pixels, the row is recomputed with the
deterministic in-repo resize path instead of silently reusing an unrelated
baseline image.

Exact mode adds stable grayscale PNG writing, raw pixel hashes, UTF-8/LF OCR
text normalization, Tesseract `5.5.1` validation, and a separate audit manifest
for timing/platform details.

For future experiments that deliberately trade run-001 compatibility for a
fully in-repo deterministic renderer or resize kernel, opt in explicitly:

```bash
.venv/bin/python -m ocr_bench all --exact \
  --deterministic-renderer \
  --deterministic-resize \
  --config benchmark.yaml \
  --output artifacts/run-deterministic
```

Changing ratios, target height, padding, fonts, or patterns is supported in
both modes. The deterministic renderer/resizer path is the one intended for new
cross-platform pixel regeneration; the `--baseline-run artifacts/run-001` path
is the one intended for reproducing the already-presented baseline.

The Windows CI workflow in `.github/workflows/windows-exact.yml` installs
Tesseract `5.5.1`, verifies committed font and tessdata hashes, runs tests, and
then executes:

```bash
python -m ocr_bench all --exact --deterministic-resize --force \
  --config benchmark.yaml \
  --output artifacts/run-exact-windows \
  --baseline-run artifacts/run-001
```

It asserts the frozen Traditional Chinese bilinear pooled CER,
`0.19117647058823528`, and uploads the Windows exact artifacts for inspection.

Install the pinned exact-mode Python stack with:

```bash
.venv/bin/pip install -r requirements-exact.txt
.venv/bin/pip install --no-deps -e .
```

Exact mode requires Tesseract `5.5.1`. The local macOS/Homebrew binary and data
hashes used while adding exact mode are:

| Artifact | SHA-256 |
| --- | --- |
| `/opt/homebrew/bin/tesseract` | `6517c9cf1b17280201af3e48880517bbfafd24b5876aacb75d5643bafff1c414` |
| `tessdata/eng.traineddata` | `7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2` |
| `tessdata/chi_tra.traineddata` | `529c5b5797d64b126065cd55f2bb4c7fd7b15790798091b1ff259941a829330b` |
| `fonts/Consolas-Regular.ttf` | `5f8d58e719a7d724be036145f506acf38b0942e253a7c331887c7056b93deac8` |
| `fonts/microsoft-jhenghei.ttf` | `03a01753d6916c13bb2d2d159ba6f858949d191059b588f143fd589bf394d101` |

Record the corresponding Windows and Linux Tesseract executable hashes from
your installed native `5.5.1` binaries before treating those platforms as exact
production baselines.

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

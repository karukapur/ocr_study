# Replicate Bilinear

Generate high-resolution synthetic source patterns:

```bash
.venv/bin/python experiments/replicate_bilinear/test_patterns/generate_test_patterns.py
```

Run the OpenCV-vs-fixed-point bilinear comparison:

```bash
.venv/bin/python experiments/replicate_bilinear/compare_bilinear.py
```

All three comparison scripts in this directory are standalone: none imports
code from another experiment script.

This reads PNG, TIFF (`.tif`/`.tiff`), and SVG files from `test_patterns/`,
normalizes them to 8-bit RGB, downsamples each color plane at the configured
ratios, compares OpenCV bilinear against `fixed_point_bilinear()`, and writes:

- `results/bilinear_comparison_metrics.csv`
- `results/summary.md`
- `results/plots/*.svg`

## SQNR Metric

The report computes signal-to-quantization-noise ratio (SQNR) using the OpenCV
output as the signal and the fixed-point-minus-OpenCV pixel difference as the
noise:

```text
e_i = fixed_i - opencv_i

P_signal = (1 / N) * sum_i(opencv_i^2)
P_noise  = (1 / N) * sum_i(e_i^2)

SQNR_dB = 10 * log10(P_signal / P_noise)
```

If `P_noise` is zero, SQNR is reported as infinity because the resized images
match exactly. If `P_signal` is zero while the error is nonzero, SQNR is reported
as negative infinity.

SVG inputs are rasterized in memory with CairoSVG on a white background. No
converted sources or resized images are saved by default. Each source pattern
and ratio is scored and its comparison arrays are discarded before processing
continues; `--write-images` only adds the resized and difference PNG outputs.

After changing resampling code in `src/ocr_bench/resample.py`, rerun:

```bash
.venv/bin/python experiments/replicate_bilinear/compare_bilinear.py
```

If only the CSV already exists and you just want to regenerate the summary/plots:

```bash
.venv/bin/python experiments/replicate_bilinear/compare_bilinear.py --reports-only
```

To also save resized outputs and absolute-difference images:

```bash
.venv/bin/python experiments/replicate_bilinear/compare_bilinear.py --write-images
```

## Floating-Point PSNR Comparisons

Compare OpenCV bilinear against the project's float32 bilinear implementation:

```bash
.venv/bin/python experiments/replicate_bilinear/compare_bilinear_floating_psnr.py
```

Each comparison script uses its built-in ratio sweep by default. Pass one
value to use the same downscale ratio for width and height:

```bash
.venv/bin/python experiments/replicate_bilinear/compare_bilinear_floating_psnr.py --ratio 2.0
```

Pass width and height ratios, in that order, to scale the dimensions
differently:

```bash
.venv/bin/python experiments/replicate_bilinear/compare_bilinear_floating_psnr.py --ratio 2.0 1.5
```

Repeat `--ratio` to run several configurations:

```bash
.venv/bin/python experiments/replicate_bilinear/compare_bilinear_floating_psnr.py \
  --ratio 2.0 \
  --ratio 2.0 1.5
```

The same syntax works with `compare_bilinear.py` and
`compare_bilinear_floating_psnr_github.py`. Ratios are downscale divisors, so
the output dimensions are approximately `source_width / width_ratio` and
`source_height / height_ratio`.

Compare OpenCV bilinear against the float64 implementation posted in OpenCV
issue #25018:

```bash
.venv/bin/python experiments/replicate_bilinear/compare_bilinear_floating_psnr_github.py
```

Both comparisons use half-pixel coordinate mapping, clamp image boundaries,
and calculate PSNR between the floating outputs before `uint8` quantization
with OpenCV's `cv2.PSNR` using a peak value of 255. `ResizeResult.pixels` remains
the quantized output used by the OCR benchmark, while
`ResizeResult.floating_pixels` exposes the experimental floating output. The
scripts write separate metrics CSV and summary files under
`results_floating_psnr/` and
`results_floating_psnr_github/`, respectively. Pass `--write-images` to either
script to also save its resized and absolute-difference images.

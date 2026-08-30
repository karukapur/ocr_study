# Replicate Bilinear

Generate high-resolution synthetic source patterns:

```bash
.venv/bin/python experiments/replicate_bilinear/test_patterns/generate_test_patterns.py
```

Run the OpenCV-vs-fixed-point bilinear comparison:

```bash
.venv/bin/python experiments/replicate_bilinear/compare_bilinear.py
```

This reads PNGs from `test_patterns/`, downsamples each pattern at the configured
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

No resized images are saved by default.

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

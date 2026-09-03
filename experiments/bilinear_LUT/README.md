# Bilinear LUT

This folder contains fixed-point bilinear coefficient lookup tables matching
`src/ocr_bench/resample.py`.

The default table, `opencv_linear_q11_coefficients.csv`, enumerates every Q11
fraction phase from `0 / 2048` through `2048 / 2048`. Each row stores the two
integer bilinear coefficients used by the fixed-point path:

```text
alpha0 = round_float32((1 - fraction) * 2048)
alpha1 = round_float32(fraction * 2048)
```

Regenerate the default table:

```bash
PYTHONPATH=src .venv/bin/python experiments/bilinear_LUT/generate_bilinear_lut.py
```

Generate concrete per-destination terms for a resize length:

```bash
PYTHONPATH=src .venv/bin/python experiments/bilinear_LUT/generate_bilinear_lut.py \
  --source-length 64 \
  --destination-length 32 \
  --axis x
```

That writes a second CSV named like
`opencv_linear_terms_x_64_to_32_shift_clamp.csv` containing destination index,
source offset, coordinate, fraction, left/right coefficients, and the OpenCV
edge valid range.

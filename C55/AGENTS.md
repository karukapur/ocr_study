# C55 maintenance requirements

- Keep this code simple and procedural: ordinary functions, explicit flow, no
  new classes, generic frameworks, or duplicated bilinear kernels.
- Keep new implementation and maintenance files under `C55/`. Do not change
  `src/ocr_bench/resample.py` for this wrapper. Bilinear must call its existing
  `fixed_point_bilinear()` per RGB channel, never floating-point bilinear or OpenCV.
- Preserve the kernel's existing floating-point setup and integer pixel arithmetic
  unless a future explicitly authorized task changes that contract. Do not claim
  an integer-only or hardware-bit-exact implementation.
- Preserve screenshot register spelling, capitalization, spaces, and punctuation.
  `Resampler: method` and `Resampler: interpolation` are software-only selectors.
- Keep documentation and its catalogue/dependencies in `docs/`; keep this file at
  the C55 root so these instructions apply to the entire folder.
- Enforce `docs/README.md`'s crop precedence, RGB uint8 shape, dimensions divisible by
  8, per-axis output/input range [1/8, 7/8], and strict truncated Q4.20 validation.
  Q4.20 increments validate dimensions; they do not drive sampling in this version.
- Nearest and translated 2×2 average use dimension-based integer rational
  coordinates, original/shift mappings, half-up rounding, and clamped neighbors.
  Average is not area resampling. Widen samples before summing.
- Coefficient selectors are zero-only inactive placeholders. Do not invent a
  bilinear bank, LUT numeric format, register address, or hardware limit. Preserve
  the documented LUT index discrepancy and distinguish facts from conventions.
- Actual future pipeline input will be YUV; current input is RGB only. Do not
  silently reinterpret YUV as RGB or add implicit color conversion. Future YUV
  support needs explicit layout, subsampling, range, and conversion decisions.
- Treat `docs/register_catalogue.json` as the reference source. Regenerate
  `docs/registers.md` and `docs/registers.xlsx` together using
  `python -m C55.generate_register_reference`.
  Keep examples and validation consistent with the catalogue. `openpyxl` remains
  a documentation-only dependency in `docs/requirements-docs.txt`.
- Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest C55/tests tests -q` from
  the repository root after changes. Check bilinear equality to the unchanged
  kernel, explicit nearest/average expected pixels, validation, CLI behavior,
  and generated reference consistency. Report limitations and test failures.

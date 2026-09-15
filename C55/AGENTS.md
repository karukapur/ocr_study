# C55 maintenance requirements

- Keep this code simple and procedural: ordinary functions, explicit flow, no
  new classes, generic frameworks, or duplicated filtering paths.
- Keep new implementation and maintenance files under `C55/`. Do not change
  `src/ocr_bench/resample.py` for this wrapper. The user-authorized 2026-09-15
  replacement uses the local row-streaming kernel for every C55 method. Keep the
  independent benchmark kernel unchanged. Do not claim hardware bit equivalence.
- Preserve screenshot register spelling, capitalization, spaces, and punctuation.
  `Resampler: method` and `Resampler: interpolation` are software-only selectors.
- Keep the usage and processing README at the C55 root. Keep supporting
  documentation and its catalogue/dependencies in `docs/`; keep this file at
  the C55 root so these instructions apply to the entire folder.
- Enforce `README.md`'s crop precedence, RGB uint8 shape, dimensions divisible by
  8, per-axis output/input range [1/8, 7/8], and strict truncated Q4.20 validation.
  Q4.20 increments drive sampling. Use Q21 coordinate state to preserve half-pixel
  initialization exactly; nearest-even phase selection must never alter that state.
- Nearest and translated 2×2 average use unquantized register-driven
  coordinates, original/shift mappings, half-up rounding, and clamped neighbors.
  Average is not area resampling. Widen samples before summing.
- Both coefficient selectors select software bilinear bank zero; reject other
  banks. Its 32 phases, seven signed 8-bit slots, scale 64, and offsets -3..3 are
  explicit model conventions. Preserve horizontal Q6 sums, round the Q12 vertical
  result nearest-even, then clamp to uint8. Do not invent register addresses or
  hardware limits. Preserve the LUT index discrepancy and distinguish facts from
  conventions. Generate the ignored local bank with
  `python -m C55.generate_coefficients --output C55/bilinear_coefficients.json`.
- Latch configuration before consuming input. Keep at most seven filtered rows;
  validate every input row, including cropped-out/trailing rows and extra input.
  Streaming output is provisional until successful iterator exhaustion.
- Actual future pipeline input will be YUV; current input is RGB only. Do not
  silently reinterpret YUV as RGB or add implicit color conversion. Future YUV
  support needs explicit layout, subsampling, range, and conversion decisions.
- Treat `docs/register_catalogue.json` as the reference source. Regenerate
  `docs/registers.md` and `docs/registers.xlsx` together using
  `python -m C55.generate_register_reference`.
  Keep examples and validation consistent with the catalogue. `openpyxl` remains
  a documentation-only dependency in `docs/requirements-docs.txt`.
- Generated coefficients, register Markdown/workbook, and `outputs/` are ignored
  by Git. Keep the generators and source catalogue tracked. Generate the bank and
  both register references before testing a fresh checkout.
- Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest C55/tests tests -q` from
  the repository root after changes (use `.venv/bin/python` in this workspace).
  Check independent scalar references, phase ties/carry/drift, bounded streaming,
  all methods, validation, CLI behavior, and generated bank/reference consistency.
  Report limitations and test failures.

# C55 crop and resampler reference

This is a software reference with C55 register names, not a verified bit-exact C55
hardware model. All new processing code is in the parent `C55/` directory. The bilinear path
calls the extensively tested `ocr_bench.resample.fixed_point_bilinear` unchanged;
nearest and translated 2×2 average are local implementations.

**The eventual pipeline input will be YUV. This version accepts only 8-bit RGB.**
YUV has different channel meanings and may have subsampled chroma. Its layout,
subsampling, range, and conversion must be specified before future integration.
The wrapper performs no color conversion and does not infer an array's color space.

## Run from the repository root

Install the existing project into your Python environment:

```sh
python -m pip install -e '.[dev]'
python -m C55.resampler --input input.png --registers C55/registers.example.json --output output.png
```

Input image mode must be `RGB`: grayscale, palette, RGBA, and other modes are
rejected instead of silently converted. Output is always lossless PNG, regardless
of the filename suffix. Decode errors, malformed configuration, and invalid frame
dimensions report an error and exit nonzero. Stream framing is not modeled.

The example crops 640×480 at `(16, 8)` and produces 320×240. Its source image must
be at least **656×488**. Both Q4.20 increments are `2097152` (2 × 2^20).

To use another method or mapping, edit the two **software-only** fields:

| Field | Values | Example default |
| --- | --- | --- |
| `Resampler: method` | `0` nearest, `1` fixed-point bilinear, `2` translated 2×2 average | `1` |
| `Resampler: interpolation` | `0` original, `1` shift | `1` |

Both mappings are available with all three methods. All 16 fields are required;
the JSON uses integer values, not strings, floats, or booleans. Unknown and
duplicate keys are errors. See [registers.md](registers.md) and
[registers.xlsx](registers.xlsx) for the complete register reference.

## Python interface

```python
import numpy as np
from PIL import Image
from C55.resampler import crop_and_resize, load_registers

registers = load_registers("C55/registers.example.json")
with Image.open("input.png") as source:
    if source.mode != "RGB":
        raise ValueError("RGB input required")
    image = np.array(source)
output = crop_and_resize(image, registers)
Image.fromarray(output).save("output.png")
```

`load_registers(path) -> dict` checks field names, integer types, and selectors.
`crop_and_resize(image, registers) -> np.ndarray` also checks geometry, dimensions,
and increments against the actual frame before processing. It accepts nonempty
`uint8` arrays of shape `(height, width, 3)` and returns an independent RGB array
without changing the source or register dictionary.

## Processing contract

1. Crop is active only when `Pipeline: Bypass crop` is `0` and `Crop: Enable crop`
   is `1`. Bypass wins over enable. Inactive crop geometry is ignored after its
   integer types are checked. Active crop requires nonnegative origins, positive
   sizes, and a rectangle wholly inside the actual input.
2. `width_in` and `height_in` must equal the effective crop dimensions, or the
   full input dimensions when crop is inactive. Both must be divisible by 8.
3. Output dimensions are explicit positive integers. For each axis,
   `input <= 8 * output <= 7 * input`; identity and enlargement are rejected.
   These are the supplied Resampler slide's limits. Odd output sizes are valid;
   8×8 → 1×1 is valid. No scale-to-output rounding API is provided.
4. Require `hfilt_tinc == (width_in << 20) // owidth` and
   `vfilt_tinc == (height_in << 20) // oheight`, in assumed unsigned Q4.20 range
   `[2^20, 2^24-1]`. Python integer arithmetic avoids validation overflow.
5. Both coefficient selectors must be zero, an **inactive placeholder**. No
   programmable LUT is executed, and zero does not identify a known bilinear bank.

The input array/file supplies the actual frame extent. Inconsistent dimensions
are rejected. Border clamping cannot conceal a short or malformed input frame.
Crop coordinates are relative to the image arriving at this block.

### Sampling

For output coordinate `u`, input width `W`, and output width `M`:

| Mapping | Source coordinate |
| --- | --- |
| Original | `x = u * W / M` |
| Shift (half-pixel) | `x = (u + 1/2) * W / M - 1/2` |

Apply the same mapping vertically. **The dimension ratios drive sampling; the
truncated Q4.20 registers are validation fields, not phase accumulators.**

- Nearest evaluates the dimension-based coordinate as an exact integer rational,
  rounds the index half up, and clamps it to the effective input edge.
- Average takes the four neighbors at `floor(x)`, `floor(x)+1`, `floor(y)`,
  `floor(y)+1`, clamps each neighbor, sums using widened integers, and computes
  `(sum + 2) // 4`. This is not a full-area filter or non-overlapping box binning.
- Bilinear runs the existing fixed-point function independently on R, G, and B.
  It retains floating-point coordinate/coefficient setup, Q11 coefficients,
  integer accumulation, clamped boundaries, OpenCV-compatible intermediate
  truncation/rounding, and its exact 2× shift special case. It calls neither
  OpenCV nor the floating-point bilinear implementation. It is not integer-only.

For example, at 2× reduction a shift coordinate starts at 0.5. Nearest selects
index 1 on each axis (half up); average uses the first 2×2 group. At fractional
ratios the two mappings can select different groups. The bilinear setup's
floating-point quantization can differ from exact rational coordinates.

## Hardware interpretation and maintenance

The [boundary document](bilinear-scaler-input-and-boundary-conditions.md) explains
proposed consistency checks and edge handling. This wrapper adopts those checks
with the stricter slide size limits; it does not rewrite the document's general
bilinear examples into hardware requirements. No undocumented maximum frame
dimension or register address is assigned.

The LUT screenshot describes seven coefficients per phase, 32 phases per set,
and 16 sets. Its final index labels disagree with those dimensions. The generated
references preserve and explain this discrepancy, byte packing, and the apparent
`hfilt_tinc` heights/widths typo. Coefficient signedness, normalization, hardware
phases, and memory endianness have not been established.

Before a C/hardware port, specify supported dimensions and choose sufficiently
wide types for scale shifts, coordinate products, and addresses. An average sum
can reach 1020. Verify hardware phase, rounding, saturation, boundaries, and
coefficient contents before claiming pixel equivalence. Do not replace the
current tested bilinear kernel silently as part of that port.

### Regenerate cross-team references

`register_catalogue.json` is the source of truth for register descriptions and
reference notes. Edit it and regenerate both outputs together:

```sh
python -m pip install -r C55/docs/requirements-docs.txt
python -m C55.generate_register_reference
```

The workbook includes Registers, LUT layout, and Notes sheets. `openpyxl` is only
needed for documentation generation and the workbook checks, not for resampling.

### Validate

From the repository root, with project and documentation dependencies installed:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest C55/tests tests -q
```

The explicit C55 path is necessary because the existing pytest configuration
discovers only `tests/`. Disabling third-party plugin autoload avoids unrelated
ROS pytest plugins in the development environment.

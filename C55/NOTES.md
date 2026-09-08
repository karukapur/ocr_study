# C55 discussion notes

> **APPEND-ONLY FILE:** Any future changes, clarifications, corrections, or new
> discussions MUST be appended to the end of this file. Do not overwrite,
> replace, delete, or rewrite existing entries. Add corrections as new dated
> entries identifying the earlier statement being corrected, preserving history.

## 2026-09-08 — Scale registers, bilinear boundaries, and configuration responsibilities

These notes preserve the substance of the discussion about
[bilinear-scaler-input-and-boundary-conditions.md](docs/bilinear-scaler-input-and-boundary-conditions.md)
and the supplied Arm table, “Table 3-47: Down scaler key parameters.” They
distinguish register descriptions, proposed model behavior, and the current
repository implementation. The discussion does not establish hardware equivalence
or authorize changes to the processing implementation.

### What T_x, T_y, and 2^20 mean

- `W = width_in`: width of the image entering the scaler.
- `H = height_in`: height of the image entering the scaler.
- `M = owidth`: output width.
- `N = oheight`: output height.
- `T_x = hfilt_tinc`: stored horizontal scale increment.
- `T_y = vfilt_tinc`: stored vertical scale increment.

The represented increments are:

```text
s_x = T_x / 2^20  ≈ W / M
s_y = T_y / 2^20  ≈ H / N
2^20 = 1,048,576
```

An increment of 2 means advancing two input pixels horizontally for each output
pixel, or two input rows vertically for each output row. These ratios are
input/output; a 2× reduction has an increment of 2, not 0.5.

The registers use 4.20 fixed-point representation: four integer bits and twenty
fractional bits. The stored value is an integer with an implied binary point.

| Represented increment | Stored integer |
| --- | ---: |
| 1.0 | 1,048,576 |
| 1.5 | 1,572,864 |
| 2.0 | 2,097,152 |

We use 2^20 because the format reserves twenty fractional bits. One stored unit
represents 2^-20, approximately 0.000000954 input pixels. Powers of two fit binary
arithmetic; integer and fractional portions can be separated with shifts and
masks. Twenty bits is a register-format design choice, not a requirement of
bilinear interpolation. Other designs can choose different precision. The table
does not explain Arm's particular engineering rationale for choosing twenty.

### Simple general-purpose bilinear behavior discussed

The suggested mathematical reference contract is dimension validation,
half-pixel mapping, bilinear interpolation, and edge replication. “General-purpose”
means mathematically defined for positive image dimensions, including one-pixel
axes; it does not mean every such size is supported by C55 or this repository.

For zero-based output position `(u, v)`, dimension-based half-pixel mapping is:

```text
x = (u + 0.5) * W / M - 0.5
y = (v + 0.5) * H / N - 0.5
```

For example, width 4 to width 2 samples input positions 0.5 and 2.5.

A simple equivalent way to implement replicated edges for bilinear is:

```text
x = clamp(x, 0, W - 1)
y = clamp(y, 0, H - 1)
x0 = floor(x); x1 = min(x0 + 1, W - 1); a = x - x0
y0 = floor(y); y1 = min(y0 + 1, H - 1); b = y - y0

top    = (1 - a) * I[y0, x0] + a * I[y0, x1]
bottom = (1 - a) * I[y1, x0] + a * I[y1, x1]
O[v, u] = (1 - b) * top + b * bottom
```

Here array indexing is `[row, column]`; apply the blend independently per channel.
This is explanatory mathematics, not a replacement for the existing kernel.
No padded image allocation is needed. Alternatively, calculate the original
coordinate's floor and weights and clamp each neighbor index, as in the boundary
document.

- At the final row or column, neighbor accesses stay within the input.
- For an input axis of length one, both neighbors select its only pixel.
- A 1×1 input produces a constant image under this mathematical contract.
- An output axis of length one samples the center of the corresponding input axis.
- Identity scaling reproduces the image, while clamping still prevents an
  unconditional zero-weight neighbor read from going out of bounds.
- Invalid dimensions, incomplete input, and out-of-frame crops must be rejected;
  edge replication must not disguise missing real input pixels.

This follows the basic half-pixel and replicated-edge sampling behavior of
OpenCV `resize(..., INTER_LINEAR)`. A usage example discussed was
`cv2.resize(image, (M, N), interpolation=cv2.INTER_LINEAR)`.
Matching the mathematical rule does not guarantee identical output bits:
coordinate precision, coefficient quantization, intermediate arithmetic, and
rounding also matter. Q4.20 scale storage alone does not ensure OpenCV equality.

References: [OpenCV resize source](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/resize.cpp)
and [OpenCV transformation documentation](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html).

### Crop origin

Crop origin `(c_x, c_y)` is the top-left position of the crop in the input image:
`c_x` counts columns from the left and `c_y` counts rows from the top.

Origin `(100, 50)` with size 640×480 keeps columns 100–739 and rows 50–529,
using zero-based indexing. Original position `(100, 50)` becomes crop position
`(0, 0)`. The scaler then receives the 640×480 crop.

```text
Block input → crop at (c_x, c_y), size W × H → scaler → output M × N
```

When crop precedes scaling, `width_in` and `height_in` describe the crop output.
With crop inactive, they describe the full image entering the scaler. Crop origin
belongs to crop configuration; it is not one of the scale increments.

### Why dimensions and increments are all supplied

The user correctly emphasized that C55 exposes the increments as writable
registers as well as exposing input and output dimensions. Software must supply
all of them. Saying “calculate the increments” means the configuring software
calculates and writes them; it does not mean C55 automatically derives them.

| Situation | Responsibility |
| --- | --- |
| Software configuring C55 | Determine actual input size, choose output size, calculate matching increments, and write all settings. |
| Model accepting supplied C55 settings | Read all values, validate consistency, and use them according to the model's documented contract. |

Under the strict truncated full-crop resizing contract:

```text
expected_T_x = floor(W * 2^20 / M)
expected_T_y = floor(H * 2^20 / N)
```

When settings are already supplied, these calculations are validation checks.
Do not silently overwrite supplied increments or change output dimensions to
make inconsistent settings agree.

The Linux C55 driver programs input dimensions from the crop, output dimensions
from the requested scale rectangle, and increments calculated using integer
division. It also selects coefficient banks based on the resize ratio.
See [mali_c55_rsz_program_resizer](https://codebrowser.dev/linux/linux/drivers/media/platform/arm/mali-c55/mali-c55-resizer.c.html#406).

### Which setting controls which responsibility?

| Register | Model variable | Conceptual responsibility |
| --- | --- | --- |
| `width_in` | W | Available input width and horizontal bounds. |
| `height_in` | H | Available input height and vertical bounds. |
| `owidth` | M | Number of output pixels per row. |
| `oheight` | N | Number of output rows. |
| `hfilt_tinc` | T_x | Horizontal input-coordinate increment in a register-driven sampling model. |
| `vfilt_tinc` | T_y | Vertical input-coordinate increment in a register-driven sampling model. |
| `hfilt_coefset` | Horizontal bank selector | Selects horizontal filter coefficients. |
| `vfilt_coefset` | Vertical bank selector | Selects vertical filter coefficients. |

These are logical responsibilities, not a verified physical ordering of C55
internal stages. A conceptual register-driven model is:

```text
M, N → enumerate output positions (u, v)
T_x, T_y and initial phases → corresponding input position (x, y)
W, H → input boundaries and neighbor handling
filter coefficients → combine samples into an output pixel
```

Its coordinate equations would be:

```text
x = phi_x + u * T_x / 2^20
y = phi_y + v * T_y / 2^20
```

For the proposed half-pixel convention, `phi_x = (s_x - 1) / 2` and
`phi_y = (s_y - 1) / 2`. The attached table does not establish actual C55 initial
phases, boundary behavior, or internal rounding. Coefficient-bank selection also
does not establish bilinear interpolation without knowing the bank contents.

The `hfilt_tinc` description in the supplied table appears to contain a typo:
the horizontal ratio should use widths, although the screenshot says heights.

### Example and conflicting settings

For 1280×720 → 640×360:

| Setting | Value |
| --- | ---: |
| `width_in` | 1280 |
| `height_in` | 720 |
| `owidth` | 640 |
| `oheight` | 360 |
| `hfilt_tinc` | 2,097,152 (2.0) |
| `vfilt_tinc` | 2,097,152 (2.0) |

The input dimensions say how much image exists. The output dimensions say how
many samples to produce. In the conceptual register-driven model, the increments
say how far apart those samples are in the input.

There is no documented conflict-resolution “priority” in the supplied table.
For example, input width 1280, output width 640, and increment 1.5 contradict the
strict full-width resizing contract. The proposed validator rejects this
configuration. We cannot conclude from this table that actual C55 rejects it,
overrides a register, or produces any particular result.

### Repository-specific clarification: current implementation differs from the conceptual model

When recording this discussion, the current [processing contract](docs/README.md)
and [maintenance instructions](AGENTS.md) were checked. They explicitly state:

> The dimension ratios drive sampling; the truncated Q4.20 registers are
> validation fields, not phase accumulators.

Therefore, the earlier discussion of `T_x` and `T_y` directly driving coordinates
describes a possible register-driven reference model, not what the current
wrapper implements. The current wrapper validates supplied increments against
dimensions and uses dimension-based sampling. Do not change that behavior based
on these notes alone.

Other current implementation constraints, distinct from universal bilinear math:

- Input is RGB `uint8`, shape `(height, width, 3)`; future YUV support is not implemented.
- Crop is active only when bypass is zero and enable is one; bypass takes precedence.
- Effective input dimensions must match the registers and be divisible by eight.
- Per-axis output/input ratio is restricted to `[1/8, 7/8]`; identity and enlargement
  are rejected. Positive odd output dimensions can be valid.
- Coefficient selectors must be zero and are inactive placeholders; zero is not
  evidence of a known bilinear coefficient bank.
- Bilinear calls the existing `ocr_bench.resample.fixed_point_bilinear` per channel.
  It retains floating-point setup, Q11 coefficients, and integer pixel arithmetic;
  it does not call OpenCV and is not claimed to be integer-only or C55-bit-exact.

This note addition changes documentation only. It does not modify any processing
contract or implementation.

# C55-style bilinear scaler: input requirements and boundary conditions

The crop/downscaler should validate its own input and user-programmed configuration, even when the ISP as a whole already knows the frame dimensions. The main question is whether the actual input, crop, output dimensions, scale increments, and sampling footprint form a supported configuration.

Missing dimension metadata is a separate issue, covered later in this document. It is not the main reason to perform block-local validation.

The supplied C55 register table explicitly includes `width_in` and `height_in`; it does not specify an automatic dimension-detection mode. The `hfilt_tinc` description appears to contain a typo: the horizontal ratio should use widths.

This document proposes a contract for a bilinear implementation that mimics this register interface. Pixel alignment and edge handling below are **proposed behavior**, because the supplied table alone does not establish Arm's exact behavior.

## Block-local sanity checks: the intended requirement

```text
Upstream image             Crop output / scaler input        Output
F_w x F_h   ---> crop --->           W x H        ---> scale ---> M x N
                 |
          origin (c_x, c_y)
```

Validate against the dimensions arriving at this block, which may differ from the original sensor dimensions. The scaler's `width_in` and `height_in` describe the crop output when cropping precedes scaling.

### Configuration checks before accepting a frame

For a crop of width W and height H within the block input:

$$
c_x\ge0,\quad c_y\ge0,\quad W>0,\quad H>0
$$

$$
c_x+W\le F_w,\qquad c_y+H\le F_h
$$

Require the configured scaler input dimensions to equal this crop size. With cropping bypassed, use the actual block input size instead. Perform the arithmetic without overflow.

For downscale-only operation:

$$
1\le M\le W,\qquad 1\le N\le H
$$

Under a truncation-based scale-register contract, require:

$$
T_x=\left\lfloor\frac{W2^{20}}M\right\rfloor,
\qquad
T_y=\left\lfloor\frac{H2^{20}}N\right\rfloor
$$

These equality checks are a proposed strict validator policy for full-crop resizing. They are not a claim that C55 hardware rejects inconsistent register writes.

Also check supported dimension ranges, scale ranges, coefficient-bank availability, and any pixel-format alignment requirements. A 4-bit bank selector being in range does not prove that its contents implement bilinear interpolation.

### Is there a universal minimum input size for bilinear?

There is no single minimum beyond a nonempty image that applies to every bilinear implementation:

| Implementation contract | Minimum-size implication |
|---|---|
| Edge replication with degenerate axes supported | A 1 x 1 input is mathematically valid; a one-pixel axis reduces interpolation to the remaining axis. |
| Four distinct in-frame neighbors required | At least 2 x 2 is necessary, but every output sampling footprint must also fit. |
| Hardware imposes line-buffer, pipeline, format, or register limits | Enforce the documented hardware minimum, even if the interpolation mathematics allows smaller inputs. |

If the software model intentionally excludes degenerate axes, W >= 2 and H >= 2 is a reasonable explicit design restriction. It must not be labeled as an established C55 minimum without further documentation.

For unconditional two-neighbor reads per axis without border extension, the minimum also depends on the requested output size, initial phase, and programmed increment:

$$
W\ge\left\lfloor\phi_x+(M-1)\frac{T_x}{2^{20}}\right\rfloor+2,
\qquad \phi_x\ge0
$$

$$
H\ge\left\lfloor\phi_y+(N-1)\frac{T_y}{2^{20}}\right\rfloor+2,
\qquad \phi_y\ge0
$$

Thus, passing a static minimum-size check does not guarantee that a requested sampling configuration is valid. Border replication or skipping zero-weight neighbors changes these footprint requirements.

### Check the arriving stream as well

Configuration validation cannot detect a producer sending fewer pixels than promised. At runtime, verify the expected valid pixels per line and lines per frame at the relevant block boundary. Report early termination, extra data, or framing mismatch according to an explicit error policy. Do not conceal missing input pixels using the normal border-extension rule.

### Examples of local validation

| Configuration | Result under the proposed contract |
|---|---|
| Block input 1920 x 1080; crop origin (1800, 0), size 200 x 100 | Reject: crop extends past the right edge. |
| Crop output 640 x 480; scaler input registers say 1280 x 720 | Reject: dimensions at the crop/scaler boundary disagree. |
| Scaler input 640 x 480; requested output 800 x 480 | Reject for a downscale-only implementation. |
| Scaler input 640 x 480; output 320 x 240; scale increments represent 1.5 | Reject: both increments should represent 2. |
| Scaler input 1 x 100; output 1 x 50 | Mathematically valid with edge replication; reject only if the model or hardware excludes one-pixel axes or another supported-range condition fails. |
| Scaler input 5 x 5; output 5 x 5; unconditional four-neighbor reads without extension | Last row/column requires an invalid neighbor read; use a defined edge policy or skip zero-weight reads. |

The original register table does not provide enough information to assign numerical C55-specific minimum dimensions. Obtain those from the full block specification or validate them against the target implementation.

## 1. Register notation and fixed-point representation

Define:

$$
W=\texttt{width\_in},\quad H=\texttt{height\_in},
\qquad M=\texttt{owidth},\quad N=\texttt{oheight}
$$

Let the integer register values be:

$$
T_x=\texttt{hfilt\_tinc},\qquad T_y=\texttt{vfilt\_tinc}
$$

They represent scale factors:

$$
s_x=\frac{T_x}{2^{20}}\approx\frac WM,
\qquad
s_y=\frac{T_y}{2^{20}}\approx\frac HN
$$

Here, a scale factor of 2 means advancing approximately two input pixels per output pixel.

### What does 2^20 mean?

$$
\boxed{2^{20}=1,048,576}
$$

It is 2 multiplied by itself 20 times. The registers use **4.20 fixed-point format**: 4 bits for the integer part and 20 bits for the fractional part. The register stores an integer; dividing that integer by 1,048,576 gives the represented scale factor.

| Desired scale factor | Integer stored in register |
|---:|---:|
| 1.0 | 1,048,576 |
| 1.5 | 1,572,864 |
| 2.0 | 2,097,152 |

For example, resizing from 1280 pixels wide to 640 pixels wide gives:

$$
s_x=\frac{1280}{640}=2,
\qquad
\texttt{hfilt\_tinc}=2\times1,048,576=2,097,152
$$

The fractional resolution is:

$$
\frac{1}{2^{20}}\approx0.0000009537
$$

## 2. What can be inferred when input dimensions are missing?

| Available input information | Suggested behavior |
|---|---|
| Image array with shape, or file with dimensions | Obtain W and H from that information and populate the registers. |
| Pixel stream with reliable line/frame boundaries | Count valid pixels per line and lines per frame; buffer if dimensions must be known before processing. |
| Output dimensions and scale increments only | Infer candidate dimensions under a specified quantization rule, then validate against the actual input. |
| Raw pixels without dimensions or framing | Reject: pixel values and total byte count do not uniquely identify the image shape. |

For example, with exact values:

$$
M=640,\quad N=360,\quad T_x=T_y=2\cdot2^{20}
$$

the implied input is 1280 × 720.

However, **fixed-point rounding matters**. If scale increments are truncated:

$$
T_x=\left\lfloor\frac{W2^{20}}M\right\rfloor
$$

then the possible integer widths satisfy:

$$
\boxed{\frac{MT_x}{2^{20}}\le W<
\frac{M(T_x+1)}{2^{20}}}
$$

Use this interval to find candidates; do not blindly truncate the product of output width and represented scale factor. If the interval contains exactly one integer, the width is uniquely recoverable under that assumption. This still does not prove that the received buffer contains that many pixels. Apply the same reasoning vertically.

The Linux C55 driver calculates increments with integer division and separately programs the scaler's input dimensions from the crop dimensions. This supports treating dimensions as explicit configuration. [Linux C55 resizer source](https://codebrowser.dev/linux/linux/drivers/media/platform/arm/mali-c55/mali-c55-resizer.c.html)

## 3. Bilinear interpolation needs a coordinate convention

For output pixel (u, v), define its location in the input:

$$
x=\phi_x+us_x,\qquad y=\phi_y+vs_y
$$

The initial phases matter. Two possible conventions are:

| Convention | Horizontal input coordinate |
|---|---|
| Start at input pixel zero | $x=us_x$ |
| Align pixel centers (half-pixel) | $x=(u+\tfrac12)s_x-\tfrac12$ |

Both use the same scale increment but can produce different images.

For a new bilinear reference model, half-pixel alignment is a reasonable explicit choice. **It cannot be claimed to match C55 from these registers alone.**

Set:

$$
i=\lfloor x\rfloor,\quad j=\lfloor y\rfloor,\qquad
a=x-i,\quad b=y-j
$$

The four neighboring samples are:

```text
(i,j)                         (i+1,j)
  A --------------------------- B
  |                             |
  |          * (x,y)            |
  |                             |
  C --------------------------- D
(i,j+1)                       (i+1,j+1)
```

Then, per color channel:

$$
\boxed{
O[u,v]=(1-a)(1-b)A+a(1-b)B+(1-a)bC+abD
}
$$

## 4. Boundary conditions

### Option A: Require valid neighbors without boundary extension

If the implementation always reads all four neighbors and provides **no boundary extension**, each output must satisfy:

$$
0\le\lfloor x\rfloor,\qquad \lfloor x\rfloor+1<W
$$

$$
0\le\lfloor y\rfloor,\qquad \lfloor y\rfloor+1<H
$$

For positive increments, checking the first and last output positions gives:

$$
\phi_x\ge0,\qquad \phi_x+(M-1)s_x<W-1
$$

$$
\phi_y\ge0,\qquad \phi_y+(N-1)s_y<H-1
$$

Provided the initial coordinates are nonnegative, the minimum input extent for unconditional four-sample reads is:

$$
\boxed{
W_{\min}=\left\lfloor\phi_x+(M-1)s_x\right\rfloor+2
}
$$

$$
\boxed{
H_{\min}=\left\lfloor\phi_y+(N-1)s_y\right\rfloor+2
}
$$

These are **minimum memory-access extents**, not necessarily the intended frame dimensions. At an exact integer coordinate, a neighbor has zero weight and may be skipped, relaxing the requirement.

### Option B: Extend the image using edge replication

A simpler general-purpose contract is **edge replication**:

$$
I_{\mathrm{ext}}(p,q)=
I\!\left[
\operatorname{clamp}(p,0,W-1),
\operatorname{clamp}(q,0,H-1)
\right]
$$

Use the extended image for all four neighbors. Conceptually:

```text
                 Actual image
              +---------------+
 a  a  a      | a    b    c    |      c  c  c
 d  d  d      | d    e    f    |      f  f  f
 g  g  g      | g    h    i    |      i  i  i
              +---------------+
                g    h    i
                g    h    i
```

The same rule extends the top edge and all four corners. It handles boundaries without reading outside the input. It requires knowing W and H, so **padding does not solve missing dimensions**.

## 5. Worked example and the identity-scaling corner case

Take one row with input width 5, output width 4, and ideal half-pixel mapping:

$$
s_x=\frac54=1.25,\qquad x(u)=(u+\tfrac12)1.25-\tfrac12
$$

| Output u | Input coordinate x | Neighbor indices | With edge replication |
|---:|---:|---|---|
| 0 | 0.125 | 0, 1 | 0, 1 |
| 1 | 1.375 | 1, 2 | 1, 2 |
| 2 | 2.625 | 2, 3 | 2, 3 |
| 3 | 3.875 | 3, 4 | 3, 4 |

No extension is needed here. But change to input width 5 and output width 5: the final coordinate is x = 4, and an unconditional read of i + 1 = 5 would be out of bounds. Its weight is zero, so either skip that read or clamp it to index 4.

**Even identity scaling needs an explicit policy for neighbor reads.**

## 6. Input constraints and corner cases

| Condition | Suggested bilinear-model behavior |
|---|---|
| Missing dimensions with no reliable inference or metadata | Reject before processing. |
| Zero/negative input or output dimensions | Reject. |
| Downscaler used for enlargement | Require output width ≤ input width and output height ≤ input height; reject enlargement unless explicitly supported. |
| Scale registers inconsistent with dimensions | Reject under the chosen quantization rule. |
| Input width or height is 1 | Edge replication makes interpolation well-defined; actual hardware may impose larger minimum dimensions. |
| Output width or height is 1 | Half-pixel mapping samples the corresponding input center; avoid formulas dividing by output dimension minus 1. |
| Odd dimensions | Valid for bilinear mathematics; impose alignment only when required by pixel format or hardware. |
| Short frame, short row, or insufficient buffer | Report malformed input; do not treat missing real pixels as border padding. |
| Very large dimensions | Use sufficiently wide arithmetic for fixed-point scale calculation, phase accumulation, and addresses. |
| Coefficient selector outside 0–15 | Reject; also require the selected bank to be defined. |

Assuming the stated format is **unsigned 4.20**, its representable positive scale range is:

$$
0<s\le16-2^{-20}
$$

For a downscale-only model:

$$
2^{20}\le T_x,T_y\le2^{24}-1
$$

That is a **format limit**, not proof that C55 supports every scale or dimension in that range.

## 7. What is still needed to match C55 exactly?

The `hfilt_coefset` and `vfilt_coefset` registers select coefficient banks; the table does not identify a bilinear bank.

To match C55 pixel-for-pixel, obtain or verify:

- Coefficient-bank contents and their interpretation.
- Initial horizontal and vertical sampling phases.
- Border-extension behavior.
- Intermediate precision, rounding, and saturation rules.
- Hardware-supported dimensions, scale limits, and pixel-format restrictions.

For an initial reference model, explicitly specify **dimension validation, half-pixel mapping, edge replication, and deterministic fixed-point rounding**, and test those choices independently of hardware equivalence.

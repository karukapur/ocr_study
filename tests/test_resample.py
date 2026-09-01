import numpy as np
import pytest

from ocr_bench.resample import (
    DEFAULT_PHASES,
    DEFAULT_TAPS,
    Q_SCALE,
    REFERENCE_PHASES,
    REFERENCE_TAPS,
    box_bin,
    coefficient_bank,
    fixed_point_bilinear,
    floating_point_bilinear,
    integer_factor,
    lanczos_resize,
    opencv_bilinear,
    output_length,
    reference_coefficient_bank,
    reference_lanczos_resize,
    resize_image,
)


def test_fractional_binning_example() -> None:
    assert integer_factor(1.625, "floor") == 1
    assert integer_factor(1.625, "ceil") == 2


def test_non_overlapping_box_average_rounds_to_nearest() -> None:
    image = np.array(
        [
            [0, 2, 10, 12],
            [4, 6, 14, 16],
            [20, 22, 30, 32],
            [24, 26, 34, 36],
        ],
        dtype=np.uint8,
    )
    expected = np.array([[3, 13], [23, 33]], dtype=np.uint8)
    np.testing.assert_array_equal(box_bin(image, 2), expected)


def test_box_bin_rejects_partial_blocks() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        box_bin(np.zeros((5, 6), dtype=np.uint8), 2)


@pytest.mark.parametrize("lobes", [2, 3])
def test_coefficient_bank_is_exactly_seven_taps_sixteen_phases(lobes: int) -> None:
    bank = coefficient_bank(scale=2.0, lobes=lobes)
    assert bank.shape == (16, 7)
    assert DEFAULT_PHASES == 16
    assert DEFAULT_TAPS == 7
    np.testing.assert_array_equal(bank.astype(np.int64).sum(axis=1), Q_SCALE)


@pytest.mark.parametrize("lobes", [2, 3])
def test_coefficient_bank_uses_configured_taps_and_phases(lobes: int) -> None:
    bank = coefficient_bank(scale=2.0, lobes=lobes, taps=5, phases=8)
    assert bank.shape == (8, 5)
    np.testing.assert_array_equal(bank.astype(np.int64).sum(axis=1), Q_SCALE)


@pytest.mark.parametrize("lobes", [2, 3])
def test_lanczos_is_identity_at_one_x(lobes: int) -> None:
    image = np.arange(12 * 18, dtype=np.uint8).reshape(12, 18)
    result = lanczos_resize(image, scale_w=1.0, lobes=lobes)
    np.testing.assert_array_equal(result.pixels, image)
    assert result.metadata["taps"] == 7
    assert result.metadata["phases"] == 16


@pytest.mark.parametrize("lobes", [2, 3])
def test_lanczos_white_extension_preserves_white(lobes: int) -> None:
    image = np.full((12, 18), 255, dtype=np.uint8)
    result = lanczos_resize(image, scale_w=2.625, lobes=lobes)
    np.testing.assert_array_equal(result.pixels, 255)


def test_output_dimensions_round_half_up() -> None:
    assert output_length(12, 1.625) == 7
    assert output_length(18, 1.625) == 11


def test_fixed_point_bilinear_is_deterministic_for_known_array() -> None:
    image = np.array(
        [
            [0, 16, 32, 48],
            [64, 80, 96, 112],
            [128, 144, 160, 176],
            [192, 208, 224, 240],
        ],
        dtype=np.uint8,
    )
    result = fixed_point_bilinear(image, 2.0)
    expected = np.array([[40, 72], [168, 200]], dtype=np.uint8)
    np.testing.assert_array_equal(result.pixels, expected)
    assert result.metadata["interpolation"] == "cv2.INTER_LINEAR deterministic clone"
    assert result.metadata["kernel"] == "opencv_inter_area_fast_2x"


def test_floating_point_bilinear_is_deterministic_for_known_array() -> None:
    image = (np.arange(6 * 8, dtype=np.uint8).reshape(6, 8) * 5).astype(np.uint8)
    result = floating_point_bilinear(image, 2.0)
    expected = np.array(
        [
            [23, 33, 43, 53],
            [103, 113, 123, 133],
            [183, 193, 203, 213],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(result.pixels, expected)
    assert result.floating_pixels is not None
    assert result.floating_pixels.dtype == np.float32
    np.testing.assert_array_equal(
        np.clip(np.floor(result.floating_pixels + np.float32(0.5)), 0, 255).astype(
            np.uint8
        ),
        result.pixels,
    )
    assert result.metadata["coefficient_precision"] == "float32"
    assert result.metadata["coordinate_mapping"] == "shift"


def test_opencv_bilinear_exposes_independent_float64_output() -> None:
    image = (np.arange(7 * 9, dtype=np.uint8).reshape(7, 9) * 3).astype(np.uint8)
    result = opencv_bilinear(image, 1.7)

    assert result.pixels.dtype == np.uint8
    assert result.floating_pixels is not None
    assert result.floating_pixels.dtype == np.float64
    assert result.floating_pixels.shape == result.pixels.shape
    assert result.metadata["floating_pixel_source"] == (
        "cv2.INTER_LINEAR on float64 input"
    )


def test_original_and_shift_bilinear_mappings_differ() -> None:
    image = (np.arange(6 * 8, dtype=np.uint8).reshape(6, 8) * 5).astype(np.uint8)
    shifted = floating_point_bilinear(image, 2.0, interpolation="shift")
    original = floating_point_bilinear(image, 2.0, interpolation="original")
    assert np.any(shifted.pixels != original.pixels)
    np.testing.assert_array_equal(
        original.pixels,
        np.array(
            [[0, 10, 20, 30], [80, 90, 100, 110], [160, 170, 180, 190]],
            dtype=np.uint8,
        ),
    )
    fixed = fixed_point_bilinear(image, 2.0, interpolation="original")
    np.testing.assert_array_equal(fixed.pixels, original.pixels)


def test_integer_floor_and_ceil_are_identical() -> None:
    image = np.arange(36, dtype=np.uint8).reshape(6, 6)
    floor = resize_image(image, 2.0, "bin_floor")
    ceiling = resize_image(image, 2.0, "bin_ceil")
    np.testing.assert_array_equal(floor.pixels, ceiling.pixels)


def test_reference_bank_has_129_slots_1024_normalized_phases() -> None:
    bank = reference_coefficient_bank(2.0)
    assert bank.shape == (REFERENCE_PHASES, REFERENCE_TAPS) == (1024, 129)
    np.testing.assert_allclose(bank.sum(axis=1), 1.0, rtol=0, atol=1e-13)
    nonzero_offsets = np.flatnonzero(np.abs(bank[1]) > 1e-12) - 64
    assert nonzero_offsets.min() < -3
    assert nonzero_offsets.max() > 3


def test_reference_bank_rejects_support_beyond_129_taps() -> None:
    with pytest.raises(ValueError, match="129-tap reference capacity"):
        reference_coefficient_bank(22.0)


def test_reference_lanczos_identity_and_constant_preservation() -> None:
    identity = np.arange(9 * 11, dtype=np.uint8).reshape(9, 11)
    np.testing.assert_array_equal(reference_lanczos_resize(identity, 1.0).pixels, identity)

    constant = np.full((8, 10), 73, dtype=np.uint8)
    result = reference_lanczos_resize(constant, 2.3)
    np.testing.assert_array_equal(result.pixels, 73)
    assert result.metadata["boundary"] == "reflect101"


def test_reflect_boundary_is_opt_in_for_benchmark_lanczos() -> None:
    image = np.zeros((8, 10), dtype=np.uint8)
    white = lanczos_resize(image, 2.3, 3)
    reflected = lanczos_resize(image, 2.3, 3, boundary="reflect101")
    assert white.metadata["boundary"] == "white"
    assert reflected.metadata["boundary"] == "reflect101"
    assert np.any(white.pixels != reflected.pixels)
    np.testing.assert_array_equal(reflected.pixels, 0)


@pytest.mark.parametrize(
    "method",
    [
        "opencv_bilinear",
        "floating_point_bilinear",
        "fixed_point_bilinear",
        "lanczos2",
        "lanczos3",
    ],
)
def test_interpolators_support_independent_axis_scales(method: str) -> None:
    image = np.arange(12 * 20, dtype=np.uint8).reshape(12, 20)
    result = resize_image(
        image,
        2.0,
        method,
        scale_h=3.0,
        interpolation="shift",
        boundary="reflect101",
    )
    assert result.pixels.shape == (4, 10)
    assert result.metadata["actual_scale_x"] == 2.0
    assert result.metadata["actual_scale_y"] == 3.0


def test_reference_lanczos_supports_independent_axis_scales() -> None:
    image = np.arange(12 * 20, dtype=np.uint8).reshape(12, 20)
    result = reference_lanczos_resize(image, 2.0, 3.0)
    assert result.pixels.shape == (4, 10)
    assert result.metadata["actual_scale_x"] == 2.0
    assert result.metadata["actual_scale_y"] == 3.0


@pytest.mark.parametrize(
    "resize",
    [opencv_bilinear, floating_point_bilinear, fixed_point_bilinear],
)
def test_missing_height_scale_falls_back_to_width_scale(resize) -> None:
    image = np.arange(12 * 20, dtype=np.uint8).reshape(12, 20)
    implicit = resize(image, 2.0)
    explicit = resize(image, 2.0, 2.0)
    np.testing.assert_array_equal(implicit.pixels, explicit.pixels)


def test_dispatch_rejects_anisotropic_box_binning() -> None:
    image = np.arange(12 * 20, dtype=np.uint8).reshape(12, 20)
    with pytest.raises(ValueError, match="does not support a separate height scale"):
        resize_image(image, 2.0, "bin_floor", scale_h=3.0)

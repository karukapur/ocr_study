import numpy as np
import pytest

from ocr_bench.resample import (
    PHASES,
    Q_SCALE,
    TAP_OFFSETS,
    box_bin,
    coefficient_bank,
    integer_factor,
    lanczos_resize,
    output_length,
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
    assert PHASES == 16
    assert len(TAP_OFFSETS) == 7
    np.testing.assert_array_equal(bank.astype(np.int64).sum(axis=1), Q_SCALE)


@pytest.mark.parametrize("lobes", [2, 3])
def test_lanczos_is_identity_at_one_x(lobes: int) -> None:
    image = np.arange(12 * 18, dtype=np.uint8).reshape(12, 18)
    result = lanczos_resize(image, ratio=1.0, lobes=lobes)
    np.testing.assert_array_equal(result.pixels, image)
    assert result.metadata["taps"] == 7
    assert result.metadata["phases"] == 16


@pytest.mark.parametrize("lobes", [2, 3])
def test_lanczos_white_extension_preserves_white(lobes: int) -> None:
    image = np.full((12, 18), 255, dtype=np.uint8)
    result = lanczos_resize(image, ratio=2.625, lobes=lobes)
    np.testing.assert_array_equal(result.pixels, 255)


def test_output_dimensions_round_half_up() -> None:
    assert output_length(12, 1.625) == 7
    assert output_length(18, 1.625) == 11


def test_integer_floor_and_ceil_are_identical() -> None:
    image = np.arange(36, dtype=np.uint8).reshape(6, 6)
    floor = resize_image(image, 2.0, "bin_floor")
    ceiling = resize_image(image, 2.0, "bin_ceil")
    np.testing.assert_array_equal(floor.pixels, ceiling.pixels)


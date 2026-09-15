"""Integer coordinate stepping and nearest-even phase selection."""


INCREMENT_SCALE = 1 << 20
COORDINATE_SCALE = 1 << 21
PHASES = 32


def round_nearest_even(numerator, denominator):
    """Round an integer ratio, including signed values, without floating point."""
    quotient, remainder = divmod(numerator, denominator)
    return quotient + int(2 * remainder > denominator
                          or (2 * remainder == denominator and quotient % 2 != 0))


def quantize_coordinate(coordinate):
    base, fraction = divmod(coordinate, COORDINATE_SCALE)
    phase = round_nearest_even(fraction * PHASES, COORDINATE_SCALE)
    carry, phase = divmod(phase, PHASES)
    return base + carry, phase


def axis_coordinates(count, increment, interpolation):
    """Yield (Q21 coordinate, quantized base, phase); rounding never changes state."""
    coordinate = increment - INCREMENT_SCALE if interpolation else 0
    for _ in range(count):
        base, phase = quantize_coordinate(coordinate)
        yield coordinate, base, phase
        coordinate += 2 * increment

"""PC tool for the software-model bilinear bank (not a hardware memory image)."""

import argparse
import json
from pathlib import Path


DEFAULT_COEFFICIENTS = Path(__file__).with_name("bilinear_coefficients.json")
TAP_OFFSETS = [-3, -2, -1, 0, 1, 2, 3]


def bilinear_bank():
    return {
        "phases": 32,
        "fraction_bits": 6,
        "tap_offsets": TAP_OFFSETS.copy(),
        "banks": {"0": [[0, 0, 0, 64 - 2 * p, 2 * p, 0, 0] for p in range(32)]},
    }


def validate_coefficients(coefficients):
    if not isinstance(coefficients, dict) or set(coefficients) != {
        "phases", "fraction_bits", "tap_offsets", "banks"
    }:
        raise ValueError("coefficients must contain phases, fraction_bits, tap_offsets, and banks")
    for name, expected in (("phases", 32), ("fraction_bits", 6)):
        if type(coefficients[name]) is not int or coefficients[name] != expected:
            raise ValueError(f"coefficient {name} must equal {expected}")
    offsets = coefficients["tap_offsets"]
    if (not isinstance(offsets, list) or offsets != TAP_OFFSETS
            or any(type(value) is not int for value in offsets)):
        raise ValueError("coefficient tap_offsets must be [-3, -2, -1, 0, 1, 2, 3]")
    banks = coefficients["banks"]
    if not isinstance(banks, dict) or set(banks) != {"0"}:
        raise ValueError("coefficients must provide only bilinear bank 0")
    bank = banks["0"]
    if not isinstance(bank, list) or len(bank) != 32:
        raise ValueError("coefficient bank must have 32 phases")
    for phase, weights in enumerate(bank):
        if not isinstance(weights, list) or len(weights) != 7:
            raise ValueError("each coefficient phase must have seven weights")
        if any(type(value) is not int or not -128 <= value <= 127 for value in weights):
            raise ValueError("coefficients must be signed 8-bit integers")
        if sum(weights) != 64:
            raise ValueError("coefficient weights must sum to 64")
        if weights != [0, 0, 0, 64 - 2 * phase, 2 * phase, 0, 0]:
            raise ValueError(f"coefficient phase {phase} must contain the specified bilinear weights")


def unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def load_coefficients(path=DEFAULT_COEFFICIENTS):
    with open(path, encoding="utf-8") as stream:
        coefficients = json.load(stream, object_pairs_hook=unique_object)
    validate_coefficients(coefficients)
    return coefficients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        args.output.write_text(json.dumps(bilinear_bank(), indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()

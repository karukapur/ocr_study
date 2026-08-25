from ocr_bench.pipeline import EXACT_IMAGE_RESULT_FIELDS, IMAGE_RESULT_FIELDS


def test_exact_image_results_exclude_timing_and_include_hashes() -> None:
    assert "ocr_seconds" in IMAGE_RESULT_FIELDS
    assert "ocr_seconds" not in EXACT_IMAGE_RESULT_FIELDS
    assert "source_pixel_sha256" in EXACT_IMAGE_RESULT_FIELDS
    assert "output_pixel_sha256" in EXACT_IMAGE_RESULT_FIELDS
    assert "ocr_stdout_sha256" in EXACT_IMAGE_RESULT_FIELDS

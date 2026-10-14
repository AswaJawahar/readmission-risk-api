import pytest

import build_metadata


@pytest.mark.skipif(not build_metadata.CSV_PATH.exists(),
                    reason="clean_diabetic_data.csv is gitignored; only available locally")
def test_committed_metadata_matches_training_data_and_model():
    expected = build_metadata.to_json(build_metadata.build())
    assert build_metadata.METADATA_PATH.read_text(encoding="utf-8") == expected

import pandas as pd

from stroke_predict.features.outputs import assert_public_feature_output, validate_feature_dictionary


def test_feature_output_rejects_path_like_values() -> None:
    frame = pd.DataFrame({"subject_id": ["STK-001"], "bad": ["private_raw_record.set"]})

    try:
        assert_public_feature_output(frame)
    except ValueError as exc:
        assert "path-like" in str(exc)
    else:
        raise AssertionError("Expected privacy rejection")


def test_feature_dictionary_requires_core_columns() -> None:
    dictionary = pd.DataFrame({"feature_name": ["x"]})

    try:
        validate_feature_dictionary(dictionary)
    except ValueError as exc:
        assert "feature_group" in str(exc)
    else:
        raise AssertionError("Expected schema rejection")

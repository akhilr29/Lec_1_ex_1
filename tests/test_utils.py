import pandas as pd
import pytest
from mypkg.utils import find_matching_columns, load_data


def test_find_matching_columns_is_case_insensitive_substring_match():
    df = pd.DataFrame(
        {
            "Seat comfort": [1, 2],
            "Food and drink": [3, 4],
            "Age": [30, 40],
        }
    )

    result = find_matching_columns(df, ["COMFORT", "food"])

    assert result == ["Seat comfort", "Food and drink"]


def test_load_data_explains_what_to_do_when_files_are_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="playground-series-s6e10"):
        load_data(tmp_path)

import pandas as pd
from mypkg.utils import find_matching_columns


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

from data_cleaning import _individual_df_cleaning
from read_data import read_data
from typing import List, Set

import pandas as pd


def non_shared_cols(dataframes: List[pd.DataFrame]) -> Set[str]:
    cols: List[Set[str]] = [set(df.columns) for df in dataframes]

    differences: List[Set[str]] = []
    for cols_a in cols:
        for cols_b in cols:
            if cols_a == cols_b:
                continue

            differences.append(cols_a.difference(cols_b))

    all_not_shared: Set[str] = set()
    for diff in differences:
        all_not_shared.update(diff)

    return all_not_shared


if __name__ == "__main__":
    data = read_data("./TOP500_files/")
    data = [_individual_df_cleaning(df) for df in data]
    print(non_shared_cols(data))

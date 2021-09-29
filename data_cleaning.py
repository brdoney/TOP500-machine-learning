import re
from typing import Any, List, Optional, Protocol, Union, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from read_data import (read_datasets, read_mas_translations,
                       read_valid_microarchitectures)


def _columns_in_frame(data: pd.DataFrame, columns: List[str]) -> Optional[str]:
    """
    Return the first of the given columns that is a valid column in the dataframe, or
    `None` if none of them are.

    :param data: the dataframe to check the columns of
    :type data: pd.DataFrame
    :param columns: the columns that we are looking for at least one of
    :type columns: List[str]
    :return: the first column that is a valid column in the dataframe
    :rtype: Optional[str]
    """
    for col in columns:
        if col in data.columns:
            return col
    return None


def _make_cols_uniform(data: pd.DataFrame) -> pd.DataFrame:
    """
    Make columns uniform in terms of the units and names. This is to counteract changes in
    the TOP500 dataset over the years.

    NOTE: There are more inconsistencies across datasets than this function fixes, because
    this function is primarily concerned with the data that we will use in the model. So,
    if the intention is to use other columns, more fixes may need to be added.

    :param data: the dataframe to pull data from
    :type data: pd.DataFrame
    :return: a copy of the dataframe with uniform columns
    :rtype: pd.DataFrame
    """
    # Modify a copy
    data = data.copy()

    # Remappings that we will be doing - not all will be necessary
    # Useful for understanding what unit conversions we will have to do though
    renaming_mapping = {
        "Rmax": "Rmax [TFlop/s]",
        "RMax": "Rmax [TFlop/s]",
        "Mflops/Watt": "Power Efficiency [GFlops/Watts]",
        "Power Effeciency [GFlops/Watts]": "Power Efficiency [GFlops/Watts]",
        "Accelerator Cores": "Accelerator/Co-Processor Cores",
        "Cores": "Total Cores",
    }

    # Convert Mflops/W -> Gflops/W
    if "Mflops/Watt" in data.columns:
        data["Mflops/Watt"] /= 1000

    # Rmax without explicit units is actually in GFlop/s, we want TFlop/s
    perf_col_label = _columns_in_frame(data, ["Rmax", "RMax"])
    perf_col_present = perf_col_label is not None
    if perf_col_present:
        data[perf_col_label] /= 1000  # type: ignore

    # We want Power Efficiency in GFlops/W, but older data might not have a
    # power efficiency column at all (so we'll calculate it); otherwise, we can
    # just rename the column later
    efficiency_cols = [
        "Mflops/Watt",
        "Power Effeciency [GFlops/Watts]",
        "Power Efficiency [GFlops/Watts]",
    ]
    efficiency_col_present = _columns_in_frame(data, efficiency_cols) is not None
    if "Power" in data.columns and not efficiency_col_present and perf_col_present:
        perf_col_label = cast(str, perf_col_label)

        # Power Efficiency [GFlops/W] = Rmax [TFlops] / Power [W]
        data["Power Efficiency [GFlops/Watts]"] = (data[perf_col_label] / data["Power"]) * (
            1 / 1000
        )  # type: ignore

    # Do the rename, now that we have done the unit conversions
    data = data.rename(columns=renaming_mapping)  # type: ignore

    # Inspecting the datasets shows that an omission of accelerator cores corresponds with
    # having no accelerator, so we can fill in 0 for all missing values (there are some
    # outliers, but this holds the vast majority of the time)
    data["Accelerator/Co-Processor Cores"] = data["Accelerator/Co-Processor Cores"].fillna(
        value=0
    )  # type: ignore

    return data


def _apply_log_transforms(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log transforms to the Rmax and Power Efficiency columns and add them into a new
    dataset. The transformed columns will be named `"Log(Rmax)"` and `Log(Efficiency)"`.

    :param data: the dataframe to pull data from
    :type data: pd.DataFrame
    :return: a copy of the dataframe with the Rmax and Power Efficiency columns log transformed
    :rtype: pd.DataFrame
    """
    rename_map = {
        "Rmax [TFlop/s]": "Log(Rmax)",
        "Power Efficiency [GFlops/Watts]": "Log(Efficiency)",
    }

    data = data.copy()
    for old_col, transformed_col in rename_map.items():
        data[transformed_col] = np.round(np.log(data[old_col]), 3)

    return data


def _create_microarchitecture_col(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create the `"Microarchitecture"` column by extracting data from each entry's processor.

    :param data: the dataframe to pull data from
    :type data: pd.DataFrame
    :raises ValueError: if a processor is not in the format that we expect
    :return: a copy of the dataframe with the `"Microarchitecture"` column added
    :rtype: pd.DataFrame
    """
    data = data.copy()

    # Translations from non-MAS names to MAS names
    translations = read_mas_translations()

    # List of names that are already MAS, so that we can skip checking every regex
    already_mas = read_valid_microarchitectures()

    # Pattern to remove numbers (make sure process is in format we expect)
    remove_numbers_at_end = re.compile(
        r"(.*) (?:(?:[0-9]|\.|-)+C)? (?:[0-9]|\.|-)+(?:G|M)Hz")

    def to_mas(row: pd.Series) -> str:
        processor_tech: str = row["Processor Technology"]

        if processor_tech in already_mas:
            return processor_tech

        # Get rid of numbers from the process name
        processor = row["Processor"]
        match = remove_numbers_at_end.match(processor)
        if match is None:
            raise ValueError(f"Processor {processor} did not match expected regex")
        processor = match.group(1)

        # Find and return the MAS name based on which regex matches the processor
        for pattern, mas_name in translations.items():
            if pattern.match(processor) is not None:
                return mas_name

        # Print out diagnostic information about the processor, since it had no match
        full = row["Processor"]
        name = row["Name"]
        year = row["Year"]
        print(f"Unknown processor: '{processor}', full name: '{full}' @ {name}, {year}")

        return "Unknown"

    # Apply the function to each row to create the column
    data["Microarchitecture"] = data.apply(to_mas, axis=1)

    return data


def _create_coprocessor_ratio_col(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create `"Co-Processor Cores to Total Cores"` columns as a ratio of the
    `"Accelerator/Co-Processor Cores"` / `"Total Cores"`.

    It is important to note that the total number of cores does not actually include the
    number of co-processor cores in its count. However, having more co-processor cores
    than CPU cores appears uncommonly, so it serves as a good way to make the number of
    co-processor cores a reasonable fraction, as standardizing or scaling would otherwise
    be difficult.

    :param data: the dataframe to pull data from
    :type data: pd.DataFrame
    :return: a copy of the dataframe with the coprocessor ration column added
    :rtype: pd.DataFrame
    """
    data = data.copy()
    data["Co-Processor Cores to Total Cores"] = (
        data["Accelerator/Co-Processor Cores"] / data["Total Cores"]
    )
    return data


def _individual_df_cleaning(data: pd.DataFrame, dependent_var: str) -> pd.DataFrame:
    """
    Prep an individual dataframe by making its columns use the units we want and have the
    right labels, apply log transforms to Rmax and Efficiency, create the
    microarchitecture column, and selet the desired dataframe columns.

    Important other steps in the cleaning process are filtering duplicates and
    standardization/normalization, but they require all of the data to be done
    successfully, so they are omitted from this function.

    :param data: the dataframe to clean
    :type data: pd.DataFrame
    :param dependent_var: the dependent var that we are interested in
    :type dependent_var: str
    :return: a copy of the dataframe, cleaned
    :rtype: pd.DataFrame
    """
    data = _make_cols_uniform(data)
    data = _apply_log_transforms(data)
    data = _create_microarchitecture_col(data)
    data = _create_coprocessor_ratio_col(data)
    data = _select_desired_cols(data, dependent_var)
    return data


def filter_duplicates(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the dataset with all duplicates filtered, keeping only the first item
    in a pair of duplicates.

    The year column is excluded in evaluation of duplicate rows. In other words, a row can
    be considered a duplicate if all of its values are the same as another's, excluding
    its year value.

    It is important to note that the "first" item in a pair of duplicates is not based on
    date or year, but rather on which row occurs earlier in the dataframe (in other words,
    the dataframe's index is used).

    :param dataframe: the dataframe to pull data from
    :type dataframe: pd.DataFrame
    :return: a copy of the dataframe, with duplicates removed
    :rtype: pd.DataFrame
    """
    rows_before = len(dataframe)
    # Don't use year because we want to filter systems that are entered multiple years
    # without changes
    excluding_cols = dataframe.columns.difference(["Year", "Date"])
    dataframe = dataframe.drop_duplicates(subset=excluding_cols)

    print(f"Filtered duplicates to go from {rows_before} rows to {len(dataframe)}")
    return dataframe


def _select_desired_cols(data: pd.DataFrame, dependent_var: str) -> pd.DataFrame:
    """
    Select the desired columns from the dataset, including the given dependent variable,
    and move them into a new dataset.

    Specifically, the resulting dataset will include the
    * Architecture
    * Microarchitecture
    * Year
    * Processor Speed (MHz)
    * Total Cores
    * Accelerator/Co-Processor Cores
    * Date (formatted like `f"{year}{month:02}"`)
    * The specified dependent variable column

    :param data: the dataframe to pull data from
    :type data: pd.DataFrame
    :param dependent_var: the dependent variable that we want included in the resulting dataset
    :type dependent_var: str
    :return: a copy of the dataframe, with only the desired columns
    :rtype: pd.DataFrame
    """
    desired_cols = [
        "Architecture",
        "Microarchitecture",
        "Year",
        "Processor Speed (MHz)",
        "Total Cores",
        "Co-Processor Cores to Total Cores",
        "Date",
        dependent_var,
    ]
    return data[desired_cols].copy()


class Transformer(Protocol):
    """
    Protocol to describe the transformers/scalers in the sklearn.preprocessing package,
    such as StandardScaler and OneHotEncoder.
    """

    def fit(self, X, y=None) -> Any:
        ...

    def transform(self, X) -> Any:
        ...

    def fit_transform(self, X, y=None, **fit_params) -> Any:
        ...


def standardize_data(
    dataframe: pd.DataFrame, dependent_var: str, scaler: Transformer, should_fit_scaler: bool
) -> pd.DataFrame:
    """
    Create and return a new dataframe using the data from the given dataframe standardized
    using the given scaler. Does not standardize the dependent variable column.

    NOTE: If `should_fit_scaler` is true, this method will fit the scaler, which modifies
    the scaler directly, so beware of any side effects this may cause

    :param dataframe: the dataframe to pull data from
    :type dataframe: pd.DataFrame
    :param dependent_var: the dependent variable to not standardize
    :type dependent_var: str
    :param scaler: the scaler to use to standardize the data
    :type scaler: Transformer
    :param should_fit_scaler: whether the scaler should be fit before transforming the data
    :type should_fit_scaler: bool
    :return: a copy of the dataframe, with its data standardized
    :rtype: pd.DataFrame
    """
    dataframe = dataframe.copy()

    # Don't standardize the dependent variable, so it can translate across data sets
    to_standardize = dataframe.columns.difference([dependent_var, "Date"])

    if should_fit_scaler:
        scaler.fit(dataframe[to_standardize])

    dataframe[to_standardize] = scaler.transform(dataframe[to_standardize])

    return dataframe


def select_past(
    data: pd.DataFrame, select_num_datasets: int, end_at_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Select the given number of datasets, working backwards in date order starting from
    end_at_date.

    If `end_at_date` is not provided or is None, we find and use the date for the latest
    dataset.

    NOTE: The given dataframe must have the "Date" column for this method to work.

    :param data: the dataset that includes the rows we are selecting based on date
    :type data: pd.DataFrame
    :param select_num_datasets: the number of datasets to include, including the data on
    `end_at_date`
    :type select_num_datasets: int
    :param end_at_date: the date to start working backwards from, defaults to None
    :type end_at_date: Optional[str], optional
    :return: the data from datasets in the indicated range
    :rtype: pd.DataFrame
    """
    valid_months = ["06", "11"]

    # Use the maximum value if not end date is provided
    if end_at_date is None:
        end_at_date = cast(str, data["Date"].max())

    selected_dates: List[str] = [end_at_date]

    month_index = valid_months.index(end_at_date[4:])
    year = int(end_at_date[:4])

    for _ in range(select_num_datasets-1):
        if month_index == 0:
            year -= 1
        month_index = (month_index - 1) % len(valid_months)

        date_str = f"{year}{valid_months[month_index]}"
        selected_dates.append(date_str)

    selected = data[data["Date"].isin(selected_dates)].copy()

    # Sorts in ascending order (earlier date first)
    sorted_df = selected.sort_values(by="Date")

    return sorted_df


def prep_dataframe(
    data: Union[pd.DataFrame, List[pd.DataFrame]], dependent_var: str
) -> pd.DataFrame:
    # If we were given a list of dataframes, combine them into one so that we can apply
    # the pre-processing steps
    if type(data) is list:
        processed = [
            _individual_df_cleaning(df, dependent_var) for df in data
        ]
        # Concatenate all the rows, ignoring the index so we don't try merging rows from each
        # dataframe at all (essentially, we're just appending them)
        df = pd.concat(processed, ignore_index=True)
    else:
        df = cast(pd.DataFrame, data)
        df = _individual_df_cleaning(df, dependent_var)

    return df


class DataCleaner(TransformerMixin, BaseEstimator):
    _categorical_cols = {
        "Architecture": ["Cluster", "MPP", "Constellations"],
        "Microarchitecture": read_valid_microarchitectures()
    }

    def __init__(self, scaler: Transformer, dependent_var: str) -> None:
        self.scaler = scaler
        self.scaler_is_fit = False
        self.dependent_var = dependent_var

        # We provide *all* possible values for categorical columns manually now, so we
        # don't have to fit later
        categories = list(DataCleaner._categorical_cols.values())
        self.one_hot_encoder = OneHotEncoder(categories=categories, drop="first")

    def fit(self, X, y=None) -> "DataCleaner":
        # Fitting shouldn't do anything, since we've already provided data
        self.one_hot_encoder.fit(X[list(DataCleaner._categorical_cols.keys())])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Remove rows with NaNs or that are duplicates of other rows
        X = X.dropna()
        X = filter_duplicates(X)

        # One hot encode
        categories = list(DataCleaner._categorical_cols.keys())
        encoded = self.one_hot_encoder.transform(X[categories]).toarray()
        encoded_col_names = self.one_hot_encoder.get_feature_names()
        encoded_df = pd.DataFrame(encoded, columns=encoded_col_names, index=X.index)
        X = pd.concat([X, encoded_df], axis="columns").drop(columns=categories)

        # Standardise the data, making sure we only fit the scaler the first time
        X = standardize_data(X, self.dependent_var, self.scaler, not self.scaler_is_fit)
        self.scaler_is_fit = True

        return X


if __name__ == "__main__":
    # TODO: Check if already_mas.txt can be subsituted for extracting values from
    #       mas_translations.csv
    # TODO: Determine whether having efficiency and Rmax with the same units will
    #       boost performance

    # Run to see the dataset in out/sample.csv
    dependent_var = "Log(Rmax)"
    cleaner = DataCleaner(RobustScaler(), dependent_var)

    all_data = read_datasets()
    data = prep_dataframe(all_data, dependent_var)
    data = select_past(data, 3)
    data = cleaner.fit_transform(data)
    data.to_csv("out/sample_data.csv")

    # print(cleaner.get_params())
    # testing = pickle.dumps(cleaner)
    # print(testing)
    # clf = pickle.loads(testing)
    # print(clf)

    # pickle.dump(cleaner, open("preprocessor.pkl", "wb"))

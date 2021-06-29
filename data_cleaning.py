import re
from typing import List, Optional, cast

import numpy as np
import pandas as pd

from read_data import read_data, read_mas_translations, read_valid_microarchitectures


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
    # TODO: Determine whether having efficiency and Rmax with the same units will
    # boost performance

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
        data[perf_col_label] /= 1000

    # We want Power Efficiency in GFlops/W, but older data might not have a
    # power efficiency column at all (so we'll calculate it); otherwise, we can
    # just rename the column later
    efficiency_cols = [
        "Mflops/Watt", "Power Effeciency [GFlops/Watts]", "Power Efficiency [GFlops/Watts]"]
    efficiency_col_present = _columns_in_frame(data, efficiency_cols) is not None
    if "Power" in data.columns and not efficiency_col_present and perf_col_present:
        perf_col_label = cast(str, perf_col_label)

        # Power Efficiency [GFlops/W] = Rmax [TFlops] / Power [W]
        data["Power Efficiency [GFlops/Watts]"] = (
            data[perf_col_label] / data["Power"]) * (1/1000)  # type: ignore

    # Do the rename, now that we have done the unit conversions
    data = data.rename(columns=renaming_mapping)  # type: ignore

    # This will select all the accelerator data, since we've done the renaming
    # For accelerator data, we have to assume an omission means 0, since data varies
    # significantly by system (can't interpolate)
    data["Accelerator/Co-Processor Cores"] = data["Accelerator/Co-Processor Cores"].fillna(
        value=0)  # type: ignore

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
        "Power Efficiency [GFlops/Watts]": "Log(Efficiency)"
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


def _individual_df_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prep an individual dataframe by making its columns use the units we want and have the
    right labels, apply log transforms to Rmax and Effeciency, and create the
    microarchitecture column.

    Important other steps in the cleaning process are filtering duplicates and
    standardization/normalization, but they require all of the data to be done
    succesfully, so they are ommited from this function.

    :param data: the dataframe to clean
    :type data: pd.DataFrame
    :return: a copy of the dataframe, cleaned
    :rtype: pd.DataFrame
    """
    data = _make_cols_uniform(data)
    data = _apply_log_transforms(data)
    data = _create_microarchitecture_col(data)
    return data


def _create_ids_column(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create the `"ID"` column, where an ID is simply each of the numerical columns in the
    dataframe concatenated string-wise.

    :param data: the dataframe to pull data from
    :type data: pd.DataFrame
    :return: a copy of the dataframe with a ID column added
    :rtype: pd.DataFrame
    """
    data = data.copy()

    # NOTE: different datasets might have different numeric columns, throwing off
    # duplicate filtering with IDs
    numeric_data = data.select_dtypes("number").dropna(axis="columns")

    def get_row_id(row: pd.Series) -> str:
        # Concatenate each of the numeric columns in the data to get the ID
        row_id = ""
        for col in numeric_data.columns:
            row_id += str(round(row[col], 3))  # type: ignore
        return row_id

    data["ID"] = numeric_data.apply(get_row_id, axis=1)

    return data


def filter_duplicates(testing: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    """
    Removes any entries in the testing dataframe that also occur in the training dataframe.

    NOTE: This should only be applied after unecessary columns/features have been removed from
    the dataset, as it will use all numeric columns when evaluating whether two rows are
    identical.

    :param testing: the dataframe to remove duplicates from
    :type testing: pd.DataFrame
    :param training: the dataframe to use as a source to check for duplicates
    :type training: pd.DataFrame
    :return: the testing dataframe with all duplicates removed
    :rtype: pd.DataFrame
    """
    # These don't modify in-place, so this won't affect our parameters
    testing = _create_ids_column(testing)
    training = _create_ids_column(training)

    # Common scenarios for duplicate entries are that one machine is added across multiple
    # years with no change in the specs, or one is entered multiple times in the same year
    training_ids = set(training["ID"])

    # Filter out anything in the list of training ids by boolean indexing
    not_in_training_ids = ~testing["ID"].isin(training_ids)
    filtered_testing = cast(pd.DataFrame, testing[not_in_training_ids])

    # Get rid of the ID column in the returnd dataframe, so that it isn't used for testing
    filtered_testing.drop(columns=["ID"])

    return filtered_testing


def one_hot_encode(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the dataset with the Architecture and Microarchitecture columns
    one-hot encoded.

    If using this function on multiple datasets, the order of the columns and the dropped
    dummy variable are not guaranteed to be the same.

    :param data: the dataframe to pull data from
    :type data: pd.DataFrame
    :return: a copy of the dataframe, with the categorical columns encoded
    :rtype: pd.DataFrame
    """
    categorical_prefixes = {"Architecture": "a", "Microarchitecture": "ma"}
    return pd.get_dummies(
        data,
        columns=categorical_prefixes.keys(),
        prefix=categorical_prefixes.values(),
        prefix_sep=":",
        drop_first=True,
    )


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
        "Accelerator/Co-Processor Cores",
        dependent_var,
    ]
    return data[desired_cols]


def _combine_raw_data(raw_dataframes: List[pd.DataFrame], dependent_var: str) -> pd.DataFrame:
    """
    Combine the raw dataframes into a single, large dataframe.

    Notably, the dataset will have the cleaned version of only the data we are interested
    in for the model, since this is necessary to make the combination of the datasets
    possible.

    :param raw_dataframes: the list of dataframes that we wish to combine into one
    :type raw_dataframes: List[pd.DataFrame]
    :param dependent_var: the dependent variable that we are interested in, so that we
    don't remove its column
    :type dependent_var: str
    :return: the dataframe that is made up of the contents of all of the raw dataframes
    :rtype: pd.DataFrame
    """
    # Concatenate all the rows, ignoring the index so we don't try merging rows from each
    # dataframe at all (essentialy, we're just appending them)
    dataframes = [_individual_df_cleaning(df) for df in raw_dataframes]
    dataframes = [_select_desired_cols(df, dependent_var) for df in dataframes]

    return pd.concat(dataframes, ignore_index=True)


def get_data(dependent_var: str) -> pd.DataFrame:
    """
    Get the dataframe of the TOP500 data we are interested in, already cleaned and one-hot
    encoded.

    :param dependent_var: the dependent varible to use, so that it is include in the dataframe
    :type dependent_var: str
    :return: the dataframe with all of the cleaned and encoded data
    :rtype: pd.DataFrame
    """
    # TODO: Standardize/normalize the data as well
    data = read_data("./TOP500_files/")
    data = _combine_raw_data(data, dependent_var)
    data = one_hot_encode(data)
    return data


if __name__ == "__main__":
    # TODO: Check for duplicates within the dataset
    # TODO: Make IDs translate better across files (use specific columns?)
    # TODO: Check if using Accelerator cores as a fraction improves performance
    # TODO: Check if already_mas.txt can be subsituted for extracting values from
    #       mas_translations.csv
    # TODO: Remove/extract the dependent variable data from the training data

    # Run to see the dataset in results.csv
    data = get_data("Log(Rmax)")
    data.to_csv("results.csv")

    # After getting data, do train/test splits and filter for duplicates

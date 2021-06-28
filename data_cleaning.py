import pathlib
from sklearn.preprocessing import OneHotEncoder
from typing import Dict, List, Optional, Tuple, cast
import numpy as np
import pandas as pd
import re
import csv


def _read_data(data_folder_path: str) -> List[pd.DataFrame]:
    # Directory to scan data files for
    top500_dir = pathlib.Path(data_folder_path)

    # Exclude the ~FILENAME file Excel generates for open files and include xls or xlsx files
    excel_file = re.compile(r"[^~].+\.xlsx?")

    all_datasets = []
    for file in top500_dir.iterdir():
        if excel_file.match(file.name) is not None:
            print(file.name)
            # This is a file we want to process, so read it to dataframe
            # `read_excel` supports PathLike objects, so ignore typing error
            curr_data = pd.read_excel(file, header=0)  # type: ignore
            all_datasets.append(curr_data)

    return all_datasets


def _columns_in_frame(data: pd.DataFrame, columns: List[str]) -> Optional[str]:
    for col in columns:
        if col in data.columns:
            return col
    return None


def make_cols_uniform(data: pd.DataFrame) -> pd.DataFrame:
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


def apply_log_transforms(data: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Rmax [TFlop/s]": "Log(Rmax)",
        "Power Efficiency [GFlops/Watts]": "Log(Efficiency)"
    }

    data = data.copy()
    for old_col, transformed_col in rename_map.items():
        data[transformed_col] = np.round(np.log(data[old_col]), 3)

    return data


def _read_mas_translations() -> Dict[re.Pattern, str]:
    lines = pathlib.Path(
        "./microarchitectures/mas_translations.csv").read_text().splitlines()

    # Filter out all comments
    reader = csv.DictReader(filter(lambda row: row[0] != '#', lines))

    # Create dict where {regex: desired name}
    non_mas_to_mas = {}
    for line in reader:
        compiled = re.compile(line["Non-MAS Regex"])
        non_mas_to_mas[compiled] = line["MAS"]
    return non_mas_to_mas


def _read_already_mas() -> List[str]:
    return pathlib.Path("./microarchitectures/already_mas.txt").read_text().splitlines()


def create_microarchitecture_col(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # Translations from non-MAS names to MAS names
    translations = _read_mas_translations()

    # List of names that are already MAS, so that we can skip checking every regex
    already_mas = _read_already_mas()

    # Pattern to remove numbers (make sure process is in format we expect)
    remove_numbers_at_end = re.compile(
        r"^(.*) (?:(?:[0-9]|\.|-)+C)? (?:[0-9]|\.|-)+G|MHz$")

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

        return "Unknown"

    # Apply the function to each row to create the column
    data["Microarchitecture"] = data.apply(to_mas, axis=1)

    return data


def _create_ids_column(data: pd.DataFrame) -> pd.DataFrame:
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
    # These don't modify in-place, so this won't affect the parameters
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


def fit_one_hot_encoder() -> OneHotEncoder:
    categorical_cols = {
        "Architecture": ["Cluster", "MPP", "Constellations"],
        "Microarchitecture": _read_already_mas()
    }

    # Construct a dataframe with all of the values in the specified cols so that we can
    # fit on it
    df = pd.DataFrame()

    num_rows = max(len(vals) for vals in categorical_cols.values())
    for col, vals in categorical_cols.items():
        num_vals = len(vals)

        # How many reps we can cleanly do, then how many are left over to reach the
        # desired num of cols
        clean_reps = num_rows // (num_vals)
        remainder = num_rows % num_vals

        # Each entry corresponds to the repetition of the value at that index in the Series
        # Cleanly repeat as many values as we can, then cover the remainder with the last val
        all_reps = [clean_reps] * num_vals
        all_reps[-1] += remainder
        vals_col = np.array(vals)
        vals_series = pd.Series(vals_col).repeat(all_reps).reset_index(drop=True)

        df[col] = vals_series

    encoder = OneHotEncoder(drop="first")

    return encoder.fit(df)


def one_hot_encode(data: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    categorical_cols = ["Architecture", "Microarchitecture"]

    # Transforming returns a csr_matrix, which needs to be turned into a dataframe
    transformed = encoder.transform(data[categorical_cols]).toarray()  # type: ignore
    category_names = encoder.get_feature_names()
    transformed_df = pd.DataFrame(transformed, index=data.index, columns=category_names)

    # Now we need to add our encoding data to the full dataframe
    all_data = pd.concat([data, transformed_df], axis="columns")
    all_data = all_data.drop(columns=categorical_cols)

    # return data
    return all_data


def _clean_data(data: pd.DataFrame):
    data = make_cols_uniform(data)
    data = apply_log_transforms(data)
    data = create_microarchitecture_col(data)
    return data


def prep_data(test_data: pd.DataFrame, training_data: pd.DataFrame, dependent_var: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    training_data = _clean_data(training_data)
    test_data = _clean_data(test_data)

    # Take the column's we'll actually be using in testing/training
    training_data = select_desired_cols(training_data, dependent_var)
    test_data = select_desired_cols(test_data, dependent_var)

    # Has to be done after selecting desired cols so that IDs translate across datasets
    test_data = filter_duplicates(test_data, training_data)

    # One-hot encode training and testing data using the same encoder, so that their
    # resulting col labels will be the same
    enc = fit_one_hot_encoder()
    training_data = one_hot_encode(training_data, enc)
    test_data = one_hot_encode(test_data, enc)

    return (test_data, training_data)


def select_desired_cols(data: pd.DataFrame, dependent_var: str) -> pd.DataFrame:
    desired_cols = ["Architecture", "Microarchitecture", "Year", "Processor Speed (MHz)",
                    "Total Cores", "Accelerator/Co-Processor Cores", dependent_var]
    return data[desired_cols]


if __name__ == "__main__":
    # TODO: Check for duplicates within the dataset
    # TODO: Make IDs translate better across files (use specific columns?)
    # TODO: Check if using Accelerator cores as a fraction improves performance
    # TODO: Check if already_mas.txt can be subsituted for extracting values from
    #       mas_translations.csv
    # TODO: Remove/extract the dependent variable data from the training data

    data = _read_data("TOP500_files")
    # testing = _clean_data(data[0])
    # training = _clean_data(data[1])
    # training = filter_duplicates(testing, training)
    # training = select_desired_cols(training, "Log(Rmax)")

    # enc = fit_one_hot_encoder(training)
    # item = one_hot_encode(training, enc)

    training = prep_data(data[0], data[1], "Log(Rmax)")[1]

    training.to_csv("results.csv")

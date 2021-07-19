import csv
import pathlib
import re
from typing import Dict, List

import pandas as pd


def read_datasets() -> List[pd.DataFrame]:
    """
    Read all of the excel files (.xlx or .xlsx) in the given folder and turn them into
    dataframes.

    This function does not walk through the given directory in search of files, so it will
    only check the given folder and not in any sub-folders the given folder contains.

    :param data_folder_path: the path to look in for Excel files
    :type data_folder_path: str
    :return: a list of the dataframes describing the data in the files
    :rtype: List[pd.DataFrame]
    """
    # Directory to scan data files for
    top500_dir = pathlib.Path("./TOP500_files/")

    # Exclude the ~FILENAME file Excel generates for open files and include xls or xlsx files
    # While also extracting the year and month data to support train on past
    excel_file = re.compile(r"TOP500_(\d+)\.xlsx?")

    all_datasets = []
    for file in top500_dir.iterdir():
        match = excel_file.match(file.name)
        if match is not None:
            print(file.name)
            # This is a file we want to process, so read it to dataframe
            # `read_excel` supports PathLike objects, so ignore typing error
            curr_data: pd.DataFrame = pd.read_excel(file, header=0)  # type: ignore
            curr_data = curr_data.assign(Date=match.group(1))
            all_datasets.append(curr_data)

    return all_datasets


def read_mas_translations() -> Dict[re.Pattern, str]:
    """
    Read file of regex patterns to valid microarchitecture labels into a dictionary and
    return it.

    :return: a dictionary of regex pattern and its corresponding microarchitecture label.
    :rtype: Dict[re.Pattern, str]
    """
    lines = pathlib.Path(
        "./microarchitectures/mas_translations.tsv").read_text().splitlines()

    # Filter out all comments
    reader = csv.DictReader(filter(lambda row: row[0] != '#', lines), delimiter="\t")

    # Create dict where {regex: desired name}
    non_mas_to_mas = {}
    for line in reader:
        compiled = re.compile(line["Non-MAS Regex"])
        non_mas_to_mas[compiled] = line["MAS"]
    return non_mas_to_mas


def read_valid_microarchitectures() -> List[str]:
    """
    Read the file of the list of valid microarchitectures and return it.

    :return: a list of valid microarchitectures
    :rtype: List[str]
    """
    return pathlib.Path("./microarchitectures/valid_mas.txt").read_text().splitlines()

import csv
import pathlib
import re
from typing import Dict, List

import pandas as pd


def read_data(data_folder_path: str) -> List[pd.DataFrame]:
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


def read_mas_translations() -> Dict[re.Pattern, str]:
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


def read_valid_microarchitectures() -> List[str]:
    return pathlib.Path("./microarchitectures/valid_mas.txt").read_text().splitlines()

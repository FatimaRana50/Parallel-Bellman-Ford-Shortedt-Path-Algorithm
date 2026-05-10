
#   Copyright (C) 2023 Polverino Alessandro
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#   Student: Polverino Alessandro 0622702352, a.polverino15@studenti.unisa.it
#   Lecturer: Prof. Francesco Moscato, fmoscato@unisa.it
#   Assignment: HPC-Parallel-Bellman-Ford-Implementation
#
#   Purpose of this file:
#   This file offers some useful functions to analyze the results of the executions.

import os
import pandas as pd

def check_file_correct(reference_file_path: str, file_to_check_path: str):
    with open(reference_file_path, 'rb') as ref_fp:
        ref_content = ref_fp.read()
        with open(file_to_check_path, 'rb') as check_fp:
            check_content = check_fp.read()
            same_size = len(ref_content) == len(check_content)
            same_content = ref_content == check_content
            return same_size and same_content
        
def get_csv_files(data_path):
    files = []
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            files.append(os.path.join(data_path, file))
    return files

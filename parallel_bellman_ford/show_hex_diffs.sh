
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
#   Useful to quickly check if two files are different in hex format (used 
#   for checking outputs).

# get args
if [ $# -ne 2 ]; then
    echo "Usage: $0 <file1> <file2>"
    exit 1
fi

# append file names to directory
file1="./data/outputs/$1"
file2="./data/outputs/$2"

# check if files exist
if [ ! -f $file1 ]; then
    echo "File $file1 does not exist"
    exit 1
fi

if [ ! -f $file2 ]; then
    echo "File $file2 does not exist"
    exit 1
fi

# get hex diffs
diff -u <(xxd $file1) <(xxd $file2)
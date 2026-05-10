/**
 * Copyright (C) 2023 Polverino Alessandro
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * 
 * Student: Polverino Alessandro 0622702352, a.polverino15@studenti.unisa.it
 * Lecturer: Prof. Francesco Moscato, fmoscato@unisa.it
 * Assignment: HPC-Parallel-Bellman-Ford-Implementation
 * 
 * Purpose of this file:
 * This file contains the headers for utility/save_results.c.
 * 
*/
#ifndef SAVE_RESULTS_H
#define SAVE_RESULTS_H

#include <stdbool.h>
#include "./graph.h"

char* get_output_filename(char* output_folder, char* program_name);

void save_exec_data(char* csv_filename, int num_threads, int num_cuda_threads, int cuda_grid_size, int cuda_block_size, double percent_on_device, int num_processes, char* program_name, graph_t* G, double program_seconds, double bf_seconds, char* output_filename);
void save_outputs(char* output_filename, int* distance, int V);

#endif // SAVE_RESULTS_H
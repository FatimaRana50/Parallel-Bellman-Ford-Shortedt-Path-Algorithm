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
 * This file offers some utility functions to save the results of the program.
 * 
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/graph.h"
#include "../include/save_results.h"

#define MAX_V_SAVE_BIN_FILE 10000

char* get_output_filename(char* output_folder, char* program_name) {
  char* output_filename = (char*) malloc(150 * sizeof(char));
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  long micros = ts.tv_sec * 1000000L + ts.tv_nsec / 1000;

  // Combine the output folder, program name, and timestamp to create the output filename
  snprintf(output_filename, 150, "%s/%s_%ld.bin", output_folder, program_name, micros);

  return output_filename;
}

void save_exec_data(char* csv_filename, int num_threads, int num_cuda_threads, int cuda_grid_size, int cuda_block_size, double percent_on_device, int num_processes, char* program_name, graph_t* G, double program_seconds, double bf_seconds, char* output_filename) {

  FILE *fp = fopen(csv_filename, "a");
  // checking if file was opened correctly
  if (fp == NULL) {
    printf("Error opening file %s. Does it exist?\n", csv_filename);
    exit(1);
  }

  char *has_neg_cycle = "false";
  if (G->has_negative_cycle) {
    has_neg_cycle = "true";
  }

  //csv format: threads_num,cuda_threads_num,processes_num,program,vertices_num,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles
  if (G->V > MAX_V_SAVE_BIN_FILE)
    fprintf(fp, "%d,%d,%d,%s,%d,%.2f,%d,%d,%f,%f,%f,,%s\n", num_threads, num_cuda_threads, num_processes, program_name, G->V, G->D, cuda_grid_size, cuda_block_size, percent_on_device, program_seconds, bf_seconds, has_neg_cycle);
  else
    fprintf(fp, "%d,%d,%d,%s,%d,%.2f,%d,%d,%f,%f,%f,%s,%s\n", num_threads, num_cuda_threads, num_processes, program_name, G->V, G->D, cuda_grid_size, cuda_block_size, percent_on_device, program_seconds, bf_seconds, output_filename, has_neg_cycle);
    
  fclose(fp);
}

void save_outputs(char* output_filename, int* distance, int V){
  if (V > MAX_V_SAVE_BIN_FILE) {
    return;
  }
  FILE *fp = fopen(output_filename, "w");
  // checking if file was opened correctly
  if (fp == NULL) {
    printf("Error opening file %s. Does it exist?\n", output_filename);
    exit(1);
  }

  // writing distance arrays to file
  for (int i = 0; i < V; i++) {
    fwrite(&distance[i], sizeof(int), 1, fp);
  }
  fclose(fp);

}
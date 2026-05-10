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
 * This is a parallel implementation of the Bellman-Ford algorithm, realized with OpenMP.
 * 
*/


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include <omp.h>
#include "../include/graph.h"
#include "../include/save_results.h"

#define PROGRAM_NAME "only_OMP_bellman-ford"

/**
 * FUNCTION PROTOTYPES
 */
void bellman_ford(graph_t* G, int* distance, int* predecessor, int source, int n_threads);

/**
 * MAIN FUNCTION
 */
int main(int argc, char** argv) {
  struct timeval program_start, program_end, bf_start, bf_end;

  gettimeofday(&program_start, NULL);

  // getting cl arguments
  if (argc < 5) {
    printf("Invalid number of arguments\n");
    printf("Usage: %s <source_filename> <csv_filename> <output_folder> <num_threads>\n", argv[0]);
    return 1;
  }

  char* source_filename = argv[1];
  char* csv_filename = argv[2];
  char* output_folder = argv[3];

  int num_threads = atoi(argv[4]);
  if (num_threads < 1) {
    printf("Number of threads must be greater than 0\n");
    printf("Usage: %s <source_filename> <csv_filename> <output_folder> <num_threads>\n", argv[0]);
    return 1;
  }
  omp_set_dynamic(0);     // Explicitly disable dynamic teams
  omp_set_num_threads(num_threads); // Use num_threads threads for all consecutive parallel regions

  if (source_filename == NULL) {
    printf("Source filename must be specified (.bin)\n");
    printf("Usage: %s <source_filename> <csv_filename> <output_folder> <num_threads>\n", argv[0]);
    return 1;
  }

  if (csv_filename == NULL) {
    printf("Result filename must be specified (.csv)\n");
    printf("Usage: %s <source_filename> <csv_filename> <output_folder> <num_threads>\n", argv[0]);
    return 1;
  }

  if (output_folder == NULL) {
    printf("Output folder must be specified\n");
    printf("Usage: %s <source_filename> <csv_filename> <output_folder> <num_threads>\n", argv[0]);
    return 1;
  }


  graph_t* G = get_empty_graph();
  // reading source file
  read_graph_from_file(G, source_filename);


  // running algorithm 
  int* distance = (int*) malloc(G->V * sizeof(int));
  int* predecessor = (int*) malloc(G->V * sizeof(int));
  
  gettimeofday(&bf_start, NULL);

  bellman_ford(G, distance, predecessor, 0, num_threads);

  gettimeofday(&bf_end, NULL);
  long bf_microseconds = (bf_end.tv_sec - bf_start.tv_sec) * 1e6 + (bf_end.tv_usec - bf_start.tv_usec);
  double bf_seconds = bf_microseconds / 1e6;

  gettimeofday(&program_end, NULL);
  long program_microseconds = (program_end.tv_sec - program_start.tv_sec) * 1e6 + (program_end.tv_usec - program_start.tv_usec);
  double program_seconds = program_microseconds / 1e6;

  char* output_filename = get_output_filename(output_folder, PROGRAM_NAME);
  save_exec_data(csv_filename, num_threads, 0, 0, 0, 0, 1, PROGRAM_NAME, G, program_seconds, bf_seconds, output_filename);
  save_outputs(output_filename, distance, G->V);

  destroy_graph(G);
  free(distance);
  free(predecessor);
  free(output_filename);
}

/**
 * FUNCTION DEFINITIONS
 */

void bellman_ford(graph_t* G, int* distance, int* predecessor, int source_vertex, int n_threads) {
  // initializing distance and predecessor arrays
  #pragma omp parallel for
  for (int i = 0; i < G->V; i++) {
    distance[i] = DISTANCE_INFINITY;
    predecessor[i] = -1;
  }
  // set distance of source to 0
  distance[source_vertex] = 0;

  int N;
  bool* relax_done = (bool*) calloc(n_threads, sizeof(bool));
  bool relax_done_global = false;

  #pragma omp parallel default(none) shared(distance, predecessor, G, n_threads, relax_done, relax_done_global) private(N)
  {
    N = G->V;
    int weights_num = N*N;
    int my_rank = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    int chunk_size = weights_num / num_threads;
    int my_start = my_rank * chunk_size;
    int my_end = my_start + chunk_size;
    if (my_rank == num_threads - 1) {
      my_end = weights_num;
    }

    #pragma omp barrier //TODO is this necessary?

    for (int n = 0; n < N-1; n++) {

      relax_done[my_rank] = false;

      for (int i = my_start; i < my_end; i++) {
        int w;
        #pragma omp atomic read
        w = G->edges[i];
        
        int idx_u = i / N;
        int idx_v = i % N;
        
        
        int d_u = distance[idx_u];
        int d_v = distance[idx_v];

        if (w == 0) continue; // skip non-existent edges
        if (idx_u == idx_v) continue; // skip self loops
        if (d_u + w < d_v) {
          #pragma omp atomic write
          distance[idx_v] = d_u + w;
          #pragma omp atomic write
          predecessor[idx_v] = idx_u;
          #pragma omp atomic write
          relax_done[my_rank] = true;
        }
        
      }

      #pragma omp barrier
      #pragma omp single
      {
        relax_done_global = false;
        for (int i = 0; i < n_threads; i++) {
          relax_done_global |= relax_done[i];
        }
      }

      #pragma omp barrier //TODO is this necessary?

      if (!relax_done_global) {
        break;
      }
    }


    #pragma omp barrier
    for (int i = my_start; i < my_end; i++) {
        int w;
        #pragma omp atomic read
        w = G->edges[i];

        int idx_u = i / N;
        int idx_v = i % N;
        
        int d_u = distance[idx_u];
        int d_v = distance[idx_v];

        if (w == 0) continue; // skip non-existent edges
        if (idx_u == idx_v) continue; // skip self loops
        if (d_u + w < d_v) {
          #pragma omp atomic write
          G->has_negative_cycle = true;
          break;
        }
        
      }

  }

}

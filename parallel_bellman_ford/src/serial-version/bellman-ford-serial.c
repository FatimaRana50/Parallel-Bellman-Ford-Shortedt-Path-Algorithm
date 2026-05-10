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
 * This is the serial version of the Bellman-Ford algorithm.
 * 
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include "../include/graph.h"
#include "../include/save_results.h"

#define PROGRAM_NAME "Bellman-Ford_serial_version"

/**
 * FUNCTION PROTOTYPES
 */
void bellman_ford_serial(graph_t* G, int* distance, int* predecessor, int source);

/**
 * MAIN FUNCTION
 */
int main(int argc, char** argv) {

  struct timeval program_start, program_end, bf_start, bf_end;

  gettimeofday(&program_start, NULL);

  // getting cl arguments
  if (argc < 4) {
    printf("Usage: %s <source_filename> <csv_filename> <output_folder>\n", argv[0]);
    return 1;
  }

  char* source_filename = argv[1];
  char* csv_filename = argv[2];
  char* output_folder = argv[3];

  if (source_filename == NULL) {
    printf("Usage: %s <source_filename> <csv_filename> <output_folder>\n", argv[0]);
    printf("Source filename must be specified (.bin)\n");
    return 1;
  }

  if (csv_filename == NULL) {
    printf("Usage: %s <source_filename> <csv_filename> <output_folder>\n", argv[0]);
    printf("Result filename must be specified (.csv)\n");
    return 1;
  }

  if (output_folder == NULL) {
    printf("Usage: %s <source_filename> <csv_filename> <output_folder>\n", argv[0]);
    printf("Output folder must be specified\n");
    return 1;
  }


  graph_t* G = get_empty_graph();
  // reading source file
  read_graph_from_file(G, source_filename);

  // running algorithm 
  int* distance = (int*) malloc(G->V * sizeof(int));
  int* predecessor = (int*) malloc(G->V * sizeof(int));
  double millis_elapsed;
  
  gettimeofday(&bf_start, NULL);

  bellman_ford_serial(G, distance, predecessor, 0);

  gettimeofday(&bf_end, NULL);
  long bf_microseconds = (bf_end.tv_sec - bf_start.tv_sec) * 1e6 + (bf_end.tv_usec - bf_start.tv_usec);
  double bf_seconds = bf_microseconds / 1e6;

  gettimeofday(&program_end, NULL);
  long program_microseconds = (program_end.tv_sec - program_start.tv_sec) * 1e6 + (program_end.tv_usec - program_start.tv_usec);
  double program_seconds = program_microseconds / 1e6;

  char* output_filename = get_output_filename(output_folder, PROGRAM_NAME);
  save_exec_data(csv_filename, 1, 0, 0, 0, 0, 1, PROGRAM_NAME, G, program_seconds, bf_seconds, output_filename);
  save_outputs(output_filename, distance, G->V);

  destroy_graph(G);
  free(distance);
  free(predecessor);
  free(output_filename);
}

/**
 * FUNCTION DEFINITIONS
 */

void bellman_ford_serial(graph_t* G, int* distance, int* predecessor, int source) {
  // initialize distance and predecessor arrays
  for (int i = 0; i < G->V; i++) {
    distance[i] = DISTANCE_INFINITY;
    predecessor[i] = -1;
  }

  // set distance of source to 0
  distance[source] = 0;

  // in each iteration, we relax all the edges
  // if no distance was updated, we can stop

  int* edges = G->edges;
  int N = G->V;

  bool relax_done;

  // repeat N-1 times
  for (int n = 0; n < N-1; n++) {
    relax_done = false;

    for (int i = 0; i < N*N; i++) {
      int u = i / N;
      int v = i % N;
      int w = edges[i];
      if (w == 0) continue; // skip non-existent edges
      if (u == v) continue; // skip self loops
      if (distance[u] + w < distance[v]) {
        distance[v] = distance[u] + w;
        predecessor[v] = u;
        relax_done = true;
      }
    }

    if (!relax_done) break;
  }

  // check for negative cycles
  for (int i = 0; i < N*N; i++) {
    int u = i / N;
    int v = i % N;
    int w = edges[i];
    if (w == 0) continue; // skip non-existent edges
    if (u == v) continue; // skip self loops
    if (distance[u] + w < distance[v]) {
      G->has_negative_cycle = true;
      return;
    }
  }
}
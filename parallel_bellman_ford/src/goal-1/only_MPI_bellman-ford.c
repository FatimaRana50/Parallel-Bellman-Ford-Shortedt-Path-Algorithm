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
 * This is a parallel implementation of the Bellman-Ford algorithm, realized with MPI.
 * 
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include "../include/graph.h"
#include "../include/save_results.h"

/**
 * it is useful to calculate the index of the weight in the global context.
 * The global idx can be used to calculate the source and destination vertex
*/
#define local_to_global_weight_idx(local_weight_idx, rank, chunk_size) (rank * chunk_size + local_weight_idx)

#define PROGRAM_NAME "only_MPI_Bellman-Ford"

typedef struct {
  int vertex_index;
  int new_distance;
  int new_predecessor;
} distance_update_t;

distance_update_t* get_dist_updates_arr(int size) {
  distance_update_t* arr = (distance_update_t*) malloc(size * sizeof(distance_update_t));
  for (int i = 0; i < size; i++) {
    arr[i].vertex_index = -1;
    arr[i].new_distance = -1;
    arr[i].new_predecessor = -1;
  }
  return arr;
}

int set_update(distance_update_t *arr, distance_update_t update){
  int v = update.vertex_index;
  if (arr[v].vertex_index == -1) {
    arr[v] = update;
    return 1;
  }
  else if (update.new_distance < arr[v].new_distance) {
    arr[v] = update;
    return 1;
  }
  return 0;
}

int main(int argc, char **argv) {

  if (argc < 4) {
    printf("Usage: %s <source_filename> <csv_filename> <output_folder>\n", argv[0]);
    return 1;
  }

  char* source_filename = argv[1];
  char* csv_filename = argv[2];
  char* output_folder = argv[3];

  if (source_filename == NULL) {
    printf("Source filename must be specified (.bin)\n");
    printf("Usage: %s <source_filename> <csv_filename> <output_folder>\n", argv[0]);
    return 1;
  }

  if (csv_filename == NULL) {
    printf("Result filename must be specified (.csv)\n");
    printf("Usage: %s <source_filename> <csv_filename> <output_folder>\n", argv[0]);
    return 1;
  }

  if (output_folder == NULL) {
    printf("Output folder must be specified\n");
    printf("Usage: %s <source_filename> <csv_filename> <output_folder>\n", argv[0]);
    return 1;
  }


  // spawn processes, get processes number and private rank
  MPI_Init(NULL, NULL);

  double program_start = MPI_Wtime();

  int world_size, my_mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_mpi_rank);

  // read graph from file
  int V;
  float D;
  bool has_negative_cycle;

  MPI_File my_fp;
  MPI_File_open(MPI_COMM_WORLD, source_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &my_fp);

  MPI_File_read(my_fp, &V, 1, MPI_INT, MPI_STATUS_IGNORE);
  MPI_File_read(my_fp, &D, 1, MPI_FLOAT, MPI_STATUS_IGNORE);
  MPI_File_read(my_fp, &has_negative_cycle, 1, MPI_C_BOOL, MPI_STATUS_IGNORE);

  // each process reads its own chunk of the file
  int weights_num = V * V;
  int chunk_size = weights_num / world_size;
  int remainder = weights_num % world_size;
  int start = my_mpi_rank * chunk_size;
  int end = start + chunk_size;
  if (my_mpi_rank == world_size - 1) {
    end += remainder;
  }
  int my_weights_num = end - start;

  int *my_edges = (int*) malloc(my_weights_num * sizeof(int));

  MPI_File_seek(my_fp, sizeof(int) + sizeof(float) + sizeof(bool) + start * sizeof(int), MPI_SEEK_SET);
  MPI_File_read(my_fp, my_edges, my_weights_num, MPI_INT, MPI_STATUS_IGNORE);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_File_close(&my_fp);

  // MPI custom type for distance update
  MPI_Datatype MPI_DISTANCE_UPDATE;
  int blocklengths[3] = {1, 1, 1};
  MPI_Aint offsets[3] = {0, sizeof(int), 2 * sizeof(int)};
  MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
  MPI_Type_create_struct(3, blocklengths, offsets, types, &MPI_DISTANCE_UPDATE);
  MPI_Type_commit(&MPI_DISTANCE_UPDATE);


  // initializing for bf algo
  int *my_distance = (int*) malloc(V * sizeof(int));
  int *my_predecessor = (int*) malloc(V * sizeof(int));

  // running algorithm
  double bf_start = MPI_Wtime();

  // BELLMAN FORD
  for (int i = 0; i < V; i++) {
    my_distance[i] = DISTANCE_INFINITY;
    my_predecessor[i] = -1;
  }

  // set distance of source to 0
  my_distance[0] = 0;

  int N = V;

  distance_update_t *my_updates = get_dist_updates_arr(V);

  for (int n = 0; n < N - 1; n++) {
    int my_relax_made_num = 0;
    // each process relaxes its own edges
    for (int i = 0; i < my_weights_num; i++) {
      int global_weight_idx = local_to_global_weight_idx(i, my_mpi_rank, chunk_size);
      int u = global_weight_idx / N;
      int v = global_weight_idx % N;
      int w = my_edges[i];
      if (w == 0) continue; // skip non-existent edges
      if (u == v) continue; // skip self loops
      if (my_distance[u] + w < my_distance[v]) {
        my_distance[v] = my_distance[u] + w;
        my_predecessor[v] = u;
        distance_update_t update;
        update.vertex_index = v;
        update.new_distance = my_distance[v];
        update.new_predecessor = my_predecessor[v];
        my_relax_made_num += set_update(my_updates, update);
        my_relax_made_num++;
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // use alltoall to broadcast updates to all processes
    // then each process updates its own distance and predecessor

    int *relax_made_per_process = (int*) malloc(world_size * sizeof(int));
    MPI_Allgather(&my_relax_made_num, 1, MPI_INT, relax_made_per_process, 1, MPI_INT, MPI_COMM_WORLD);

    int tot_relax_made = 0;
    int *updates_per_process = (int*) malloc(world_size * sizeof(int));
    for (int i = 0; i < world_size; i++) {
      tot_relax_made += relax_made_per_process[i];
      updates_per_process[i] = V;
    }

    if(tot_relax_made == 0) {
      free(relax_made_per_process);
      free(updates_per_process);
      break;
    }

    distance_update_t *all_updates = (distance_update_t*) malloc(V * world_size * sizeof(distance_update_t));
    int *displs = (int*) malloc(world_size * sizeof(int));
    displs[0] = 0;
    for (int i = 1; i < world_size; i++) {
      displs[i] = displs[i-1] + updates_per_process[i-1];
    }

    // int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    //          void *recvbuf, const int recvcounts[], const int displs[],
    //          MPI_Datatype recvtype, MPI_Comm comm)

    MPI_Allgatherv(my_updates, V, MPI_DISTANCE_UPDATE, all_updates, updates_per_process, displs, MPI_DISTANCE_UPDATE, MPI_COMM_WORLD);

    for (int i = 0; i < V*world_size; i++) {
      int v = all_updates[i].vertex_index;
      if (v == -1) continue;
      if (all_updates[i].new_distance < my_distance[v]) {
        my_distance[v] = all_updates[i].new_distance;
        my_predecessor[v] = all_updates[i].new_predecessor;
      }
    }

    free(all_updates);
    free(relax_made_per_process);
    free(updates_per_process);
    free(displs);
  }

    free(my_updates);

  MPI_Barrier(MPI_COMM_WORLD);

  // check for negative cycles
  for (int i = 0; i < my_weights_num; i++) {
    int global_weight_idx = local_to_global_weight_idx(i, my_mpi_rank, chunk_size);
    int u = global_weight_idx / N;
    int v = global_weight_idx % N;
    int w = my_edges[i];
    if (w == 0) continue; // skip non-existent edges
    if (u == v) continue; // skip self loops
    if (my_distance[u] + w < my_distance[v]) {
      has_negative_cycle = true;
      break;
    }
  }

  free(my_edges);

  // reduce has negative cycle to rank 0
  int has_neg_cycle_result = 0;
  MPI_Reduce(&has_negative_cycle, &has_neg_cycle_result, 1, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);

  if (my_mpi_rank == 0) {
    has_negative_cycle = has_neg_cycle_result;
    double bf_end = MPI_Wtime();
    double bf_seconds = bf_end - bf_start;
    double program_end = MPI_Wtime();
    double program_seconds = program_end - program_start;

    char *output_filename = get_output_filename(output_folder, PROGRAM_NAME);
    save_outputs(output_filename, my_distance, V);
    graph_t* G = get_empty_graph();
    G->D = D;
    G->V = V;
    G->has_negative_cycle = has_negative_cycle;
    save_exec_data(csv_filename, 0, 0, 0, 0, 0, world_size, PROGRAM_NAME, G, program_seconds, bf_seconds, output_filename);
    free(output_filename);
  }

  free(my_distance);
  free(my_predecessor);
  MPI_Type_free(&MPI_DISTANCE_UPDATE);
  MPI_Finalize();

  return 0;
}
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
 * This is one of the two implementation required by the assignment. It is
 * a parallel implementation of the Bellman-Ford algorithm, realized using
 * both MPI and OpenMP.
 * 
*/


#include <mpi.h>
#include <omp.h>
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
#define mpi_local_to_global_weight_idx(local_weight_idx, rank, chunk_size) (rank * chunk_size + local_weight_idx)

#define PROGRAM_NAME "GOAL_1_Bellman-Ford"

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

  if (argc < 5) {
    printf("Usage: %s <source_filename> <csv_filename> <output_folder> <num_threads> \n", argv[0]);
    return 1;
  }

  char* source_filename = argv[1];
  char* csv_filename = argv[2];
  char* output_folder = argv[3];

  int num_threads = atoi(argv[4]);
  if (num_threads < 1) {
    printf("Number of threads must be greater than 0\n");
    printf("Usage: %s <source_filename> <csv_filename> <output_folder> <num_threads> \n", argv[0]);
    return 1;
  }
  if (source_filename == NULL) {
    printf("Source filename must be specified (.bin)\n");
    printf("Usage: %s <source_filename> <csv_filename> <output_folder> <num_threads> \n", argv[0]);
    return 1;
  }

  if (csv_filename == NULL) {
    printf("Result filename must be specified (.csv)\n");
    printf("Usage: %s <source_filename> <csv_filename> <output_folder> <num_threads> \n", argv[0]);
    return 1;
  }

  if (output_folder == NULL) {
    printf("Output folder must be specified\n");
    printf("Usage: %s <source_filename> <csv_filename> <output_folder> <num_threads> \n", argv[0]);
    return 1;
  }

  omp_set_dynamic(0);     // Explicitly disable dynamic teams
  omp_set_num_threads(num_threads); // Use num_threads threads for all consecutive parallel regions


  // spawn processes, get processes number and private rank
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided < MPI_THREAD_FUNNELED) {
    fprintf(stderr, "MPI_Init_Thread level insufficient (%d < %d)\n", provided, MPI_THREAD_FUNNELED);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  double program_start = MPI_Wtime();

  int world_size, my_mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_mpi_rank);

  // read graph from file
  int V;
  float D;
  bool has_negative_cycle;

  MPI_File my_mpi_fp;
  MPI_File_open(MPI_COMM_WORLD, source_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &my_mpi_fp);

  MPI_File_read(my_mpi_fp, &V, 1, MPI_INT, MPI_STATUS_IGNORE);
  MPI_File_read(my_mpi_fp, &D, 1, MPI_FLOAT, MPI_STATUS_IGNORE);
  MPI_File_read(my_mpi_fp, &has_negative_cycle, 1, MPI_C_BOOL, MPI_STATUS_IGNORE);

  // each process reads its own chunk of the file
  int weights_num = V * V;
  int mpi_chunk_size = weights_num / world_size;
  int remainder = weights_num % world_size;
  int start = my_mpi_rank * mpi_chunk_size;
  int end = start + mpi_chunk_size;
  if (my_mpi_rank == world_size - 1) {
    end += remainder;
  }
  int my_mpi_weights_num = end - start;

  int *my_mpi_edges = (int*) malloc(my_mpi_weights_num * sizeof(int));

  MPI_File_seek(my_mpi_fp, sizeof(int) + sizeof(float) + sizeof(bool) + start * sizeof(int), MPI_SEEK_SET);
  MPI_File_read(my_mpi_fp, my_mpi_edges, my_mpi_weights_num, MPI_INT, MPI_STATUS_IGNORE);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_File_close(&my_mpi_fp);

  // MPI custom type for distance update
  MPI_Datatype MPI_DISTANCE_UPDATE;
  int blocklengths[3] = {1, 1, 1};
  MPI_Aint offsets[3] = {0, sizeof(int), 2 * sizeof(int)};
  MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
  MPI_Type_create_struct(3, blocklengths, offsets, types, &MPI_DISTANCE_UPDATE);
  MPI_Type_commit(&MPI_DISTANCE_UPDATE);


  // initializing for bf algo
  int *my_mpi_distance = (int*) malloc(V * sizeof(int));
  int *my_mpi_predecessor = (int*) malloc(V * sizeof(int));

  // running algorithm
  double bf_start = MPI_Wtime();

  #pragma omp parallel for
  for (int i = 0; i < V; i++) {
    my_mpi_distance[i] = DISTANCE_INFINITY;
    my_mpi_predecessor[i] = -1;
  }

  my_mpi_distance[0] = 0;

  bool relax_done_global = false;
  int my_mpi_relax_made_num = 0;
  distance_update_t *my_mpi_updates = get_dist_updates_arr(V);

  #pragma omp parallel default(shared) shared(my_mpi_edges, my_mpi_distance, my_mpi_predecessor, relax_done_global, my_mpi_updates, my_mpi_relax_made_num, my_mpi_weights_num, start, V, num_threads, MPI_DISTANCE_UPDATE, has_negative_cycle, world_size, my_mpi_rank) num_threads(num_threads)
  {
    int my_omp_rank = omp_get_thread_num();
    int omp_size = omp_get_num_threads();
    int omp_chunk_size = my_mpi_weights_num / omp_size;
    int chunk_remainder = my_mpi_weights_num % omp_size;
    int my_start = my_omp_rank * omp_chunk_size;
    int my_end = my_start + omp_chunk_size;
    if (my_omp_rank == omp_size - 1) {
      // the last rank could have more vertices to process than the others
      // since int division truncates the result, we add the remainder to the last rank
      my_end += chunk_remainder;
    }
    int N = V;

    #pragma omp barrier

    // BEGIN N-1 REPS
    for (int n = 0; n < N-1; n++) {
      #pragma omp atomic write
      my_mpi_relax_made_num = 0;
      
      #pragma omp barrier

      // relax and add to updates
      for (int i = my_start; i < my_end; i++) {
        // find the global index of the weight
        int w, d_u, d_v;
        #pragma omp atomic read
        w = my_mpi_edges[i];
        int idx_u = (i + start) / V;
        int idx_v = (i + start) % V;
        #pragma omp atomic read
        d_u = my_mpi_distance[idx_u];
        #pragma omp atomic read
        d_v = my_mpi_distance[idx_v];

        if (w == 0) continue; // skip non-existent edges
        if (idx_u == idx_v) continue; // skip self loops
        if (d_u + w < d_v) {
          distance_update_t update; 
          update.vertex_index = idx_v;
          update.new_distance = d_u + w;
          update.new_predecessor = idx_u;
          #pragma omp critical
          {
            my_mpi_distance[idx_v] = d_u + w;
            my_mpi_predecessor[idx_v] = idx_u;
            my_mpi_relax_made_num += set_update(my_mpi_updates, update);
          }
        }
      }

      #pragma omp barrier

      #pragma omp master
      {
        // BEGIN master thread uses MPI to synchronize results
        
        // SYNC UPDATES

        // get the number of updates from each process
        int *relax_made_per_process = (int*) malloc(world_size * sizeof(int));
        MPI_Allgather(&my_mpi_relax_made_num, 1, MPI_INT, relax_made_per_process, 1, MPI_INT, MPI_COMM_WORLD);
        int tot_relax_made = 0;
        int *updates_per_process = (int*) malloc(world_size * sizeof(int));
        for (int i = 0; i < world_size; i++) {
          tot_relax_made += relax_made_per_process[i];
          updates_per_process[i] = V;
        }

        if (tot_relax_made != 0) {
          // gather all updates
          distance_update_t *all_updates = (distance_update_t*) malloc(V * world_size * sizeof(distance_update_t));
          int *displs = (int*) malloc(world_size * sizeof(int));
          displs[0] = 0;
          for (int i = 1; i < world_size; i++) {
            displs[i] = displs[i-1] + updates_per_process[i-1];
          }

          // int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
          //          void *recvbuf, const int recvcounts[], const int displs[],
          //          MPI_Datatype recvtype, MPI_Comm comm)

          MPI_Allgatherv(my_mpi_updates, V, MPI_DISTANCE_UPDATE, all_updates, updates_per_process, displs, MPI_DISTANCE_UPDATE, MPI_COMM_WORLD);

          for (int i = 0; i < V*world_size; i++) {
            int v = all_updates[i].vertex_index;
            if (v == -1) continue; // not updated by process
            if (all_updates[i].new_distance < my_mpi_distance[v]) {
              my_mpi_distance[v] = all_updates[i].new_distance;
              my_mpi_predecessor[v] = all_updates[i].new_predecessor;
            }
          }
          relax_done_global = true;
          free(all_updates);
          free(displs);
        }
        else {
          relax_done_global = false;
        }
        free(updates_per_process);
        free(relax_made_per_process);
        // END master thread syncronization
      }

      #pragma omp barrier // let all threads wait for the master thread to finish synchronization


      if (!relax_done_global) {
        break;
      }
    }
  
    // END N-1 REPS

    

    // check negative cycles
    for (int i = my_start; i < my_end; i++) {
      // find the global index of the weight
      int w, d_u, d_v;
      #pragma omp atomic read
      w = my_mpi_edges[i];
      int idx_u = (i + start) / V;
      int idx_v = (i + start) % V;
      #pragma omp atomic read
      d_u = my_mpi_distance[idx_u];
      #pragma omp atomic read
      d_v = my_mpi_distance[idx_v];

      if (w == 0) continue; // skip non-existent edges
      if (idx_u == idx_v) continue; // skip self loops
      if (d_u + w < d_v) {
        #pragma omp atomic write
        has_negative_cycle = true;
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &has_negative_cycle, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  free(my_mpi_edges);
  free(my_mpi_updates);


  if (my_mpi_rank == 0) {
    double bf_end = MPI_Wtime();
    double bf_seconds = bf_end - bf_start;
    double program_end = MPI_Wtime();
    double program_seconds = program_end - program_start;

    char *output_filename = get_output_filename(output_folder, PROGRAM_NAME);
    save_outputs(output_filename, my_mpi_distance, V);
    graph_t* G = get_empty_graph();
    G->D = D;
    G->V = V;
    G->has_negative_cycle = has_negative_cycle;
    save_exec_data(csv_filename, num_threads, 0, 0, 0, 0, world_size, PROGRAM_NAME, G, program_seconds, bf_seconds, output_filename);
    free(output_filename);
  }

  free(my_mpi_distance);
  free(my_mpi_predecessor);
  MPI_Type_free(&MPI_DISTANCE_UPDATE);
  MPI_Finalize();

  return 0;
}
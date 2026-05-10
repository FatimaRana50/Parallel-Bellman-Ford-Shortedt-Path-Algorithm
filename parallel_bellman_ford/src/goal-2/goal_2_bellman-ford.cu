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
 * both CUDA and OpenMP.
 * 
*/


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
extern "C" {
#include "../include/graph.h"
#include "../include/save_results.h"
}

#define PROGRAM_NAME "GOAL_2_bellman_ford"
/**
 * NVIDIA 1050Ti - CC 6.1 - PASCAL
 * SMs: 56
 * CUDA Cores/SM : 64
 * Threads / Warp : 32
 * Warps / SM : 64
 * Threads / SM : 2048
 * Max Thread Blocks / SM : 32
 * 32 bit reg. / SM : 65536
 * Max threads / thread block : 1024
 * Shared mem size cfg: 64KB
*/

#define CUDA_CHECK(X) {\
  cudaError_t e = X;\
  if (e != cudaSuccess) {\
    printf("CUDA error %d \"%s\" at %s:%d\n", e, cudaGetErrorString(e), __FILE__, __LINE__);\
    exit(1);\
  }\
}

__device__ int d_relax_done = 0;

__global__ void relax(int *edges, int *dist, int V, int weight_N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= weight_N) return;

  int u = i / V;
  int v = i % V;
  int w = edges[i];

  if (w == 0) return;
  if (u == v) return;

  int new_dist = dist[u] + w;

  int old_dist = atomicMin(&dist[v], new_dist);

  if (old_dist > new_dist)
    atomicExch(&d_relax_done, 1);
}

int main(int argc, char **argv){

  cudaEvent_t program_start, program_stop;
  cudaEventCreate(&program_start);
  cudaEventCreate(&program_stop);
  cudaEventRecord(program_start, 0);

  // getting cl arguments
  if (argc < 6) {
    printf("Invalid number of arguments\n");
    printf("Usage: %s <source_filename> <csv_filename> <output_folder> <threads_num> <device_data_proportion>\n", argv[0]);
    return 1;
  }

  char* source_filename = argv[1];
  char* csv_filename = argv[2];
  char* output_folder = argv[3];
  int num_threads = atoi(argv[4]);
  double device_data_propotion = atof(argv[5]);

  if (source_filename == NULL) {
    printf("Source filename must be specified (.bin)\n");
    printf("Usage: %s <source_filename> <csv_filename> <output_folder> <threads_num> <device_data_proportion>\n", argv[0]);
    return 1;
  }

  if (csv_filename == NULL) {
    printf("Result filename must be specified (.csv)\n");
    printf("Usage: %s <source_filename> <csv_filename> <output_folder> <threads_num> <device_data_proportion>\n", argv[0]);
    return 1;
  }

  if (output_folder == NULL) {
    printf("Output folder must be specified\n");
    printf("Usage: %s <source_filename> <csv_filename> <output_folder> <threads_num> <device_data_proportion>\n", argv[0]);
    return 1;
  }
  if (num_threads < 1) {
    printf("Number of threads must be greater than 0\n");
    printf("Usage: %s <source_filename> <csv_filename> <output_folder> <threads_num> <device_data_proportion>\n", argv[0]);
    return 1;
  }
  if (device_data_propotion < 0 || device_data_propotion > 1) {
    printf("Device data proportion must be between 0 and 1\n");
    printf("Usage: %s <source_filename> <csv_filename> <output_folder> <threads_num> <device_data_proportion>\n", argv[0]);
    return 1;
  }
  omp_set_dynamic(0);     // Explicitly disable dynamic teams
  omp_set_num_threads(num_threads); // Use num_threads threads for all consecutive parallel regions

  graph_t* G = get_empty_graph();
  // reading source file
  read_graph_from_file(G, source_filename);

  cudaError_t err;

  int* d_edges, *d_dist;

  int device_in_count = G->V*G->V * device_data_propotion;
  int device_in_size = device_in_count*sizeof(int);


  int host_in_count = G->V*G->V - device_in_count;
  int host_start_idx = device_in_count;


  CUDA_CHECK(cudaMalloc(&d_edges, device_in_size));
  CUDA_CHECK(cudaMemcpy(d_edges, G->edges, device_in_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_dist, G->V*sizeof(int)));

  int* dist = (int*)malloc(G->V*sizeof(int));
  int* predc = (int*)malloc(G->V*sizeof(int));
  int* device_out_dist = (int*)malloc(G->V*sizeof(int));

  #pragma omp parallel for
  for (int i = 0; i < G->V; i++) {
    dist[i] = DISTANCE_INFINITY;
    predc[i] = -1;
  }

  dist[0] = 0;

  cudaEvent_t bf_start, bf_stop;
  cudaEventCreate(&bf_start);
  cudaEventCreate(&bf_stop);
  cudaEventRecord(bf_start, 0);

  dim3 block(32);
  dim3 grid((device_in_count-1)/block.x+1);

  CUDA_CHECK(cudaMemcpy(d_dist, dist, G->V*sizeof(int), cudaMemcpyHostToDevice));
  for (int i = 0; i < G->V-1; i++) {
    relax<<<grid, block>>>(d_edges, d_dist, G->V, device_in_count);
    int local_relax_done = 0;
    #pragma omp parallel
    {
      int num_threads = omp_get_num_threads();
      int my_rank = omp_get_thread_num();
      int chunk_size = host_in_count / num_threads;
      int my_start = host_start_idx + my_rank * chunk_size;
      int my_end = my_start + chunk_size;
      if (my_rank == num_threads - 1) my_end = G->V * G->V;

      for (int j = my_start; j < my_end; j++) {
        int u = j / G->V;
        int v = j % G->V;
        int w;
        w = G->edges[j];

        if (w == 0) continue;
        if (u == v) continue;

        int new_dist = dist[u] + w;
        if (dist[v] > new_dist) {
          dist[v] = new_dist;
          predc[v] = u;
          local_relax_done = 1;
        }
      }
    }
    err = cudaGetLastError();
    CUDA_CHECK(err);

    cudaMemcpy(device_out_dist, d_dist, G->V*sizeof(int), cudaMemcpyDeviceToHost);

    #pragma omp parallel for
    for (int j = 0; j < G->V; j++) {
      if (device_out_dist[j] < dist[j]) {
        dist[j] = device_out_dist[j];
      }
    }

    CUDA_CHECK(cudaMemcpy(d_dist, dist, G->V*sizeof(int), cudaMemcpyHostToDevice));

    int relax_done = 0;
    cudaMemcpyFromSymbol(&relax_done, d_relax_done, sizeof(int), 0, cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    CUDA_CHECK(err);

    relax_done |= local_relax_done;
    if (!relax_done) break;

    // set relax_done to 0
    relax_done = 0;
    cudaMemcpyToSymbol(d_relax_done, &relax_done, sizeof(int), 0, cudaMemcpyHostToDevice);
  }

  

  relax<<<grid, block>>>(d_edges, d_dist, G->V, device_in_count);
  err = cudaGetLastError();
  CUDA_CHECK(err);

  int relax_done = 0;
  cudaMemcpyFromSymbol(&relax_done, d_relax_done, sizeof(int), 0, cudaMemcpyDeviceToHost);
  err = cudaGetLastError();
  CUDA_CHECK(err);

  if (relax_done) {
    G->has_negative_cycle = true;
  }


  cudaEventRecord(bf_stop, 0);
  cudaEventSynchronize(bf_stop);
  float elapsed;
  cudaEventElapsedTime(&elapsed, bf_start, bf_stop);
  elapsed = elapsed/1000.f; // convert to seconds
  cudaEventDestroy(bf_start);
  cudaEventDestroy(bf_stop);

  cudaEventRecord(program_stop, 0);
  cudaEventSynchronize(program_stop);
  float total_elapsed;
  cudaEventElapsedTime(&total_elapsed, program_start, program_stop);
  total_elapsed = total_elapsed/1000.f; // convert to seconds


  char *output_filename = get_output_filename(output_folder, PROGRAM_NAME);
  save_outputs(output_filename, dist, G->V);
  //G->has_negative_cycle = has_negative_cycle;
  save_exec_data(csv_filename, num_threads, block.x*grid.x, grid.x, block.x, device_data_propotion, 1, PROGRAM_NAME, G, total_elapsed, elapsed, output_filename);
  free(output_filename);
  cudaFree(d_edges);
  cudaFree(d_dist);
  free(dist);
  free(predc);
  free(device_out_dist);


  return 0;
}

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
 * This is a CUDA implementation of Bellman-Ford algorithm.
 * 
*/


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
extern "C" {
#include "../include/graph.h"
#include "../include/save_results.h"
}

#define PROGRAM_NAME "ONLY_CUDA_bellman_ford"
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

  // getting cl arguments
  if (argc < 4) {
    printf("Invalid number of arguments\n");
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

  graph_t* G = get_empty_graph();
  // reading source file
  read_graph_from_file(G, source_filename);

  cudaError_t err;

  int* d_edges, *d_dist;
  int in_count = G->V*G->V;
  int in_size = in_count*sizeof(int);

  CUDA_CHECK(cudaMalloc(&d_edges, in_size));
  CUDA_CHECK(cudaMemcpy(d_edges, G->edges, in_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_dist, G->V*sizeof(int)));

  int* dist = (int*)malloc(G->V*sizeof(int));
  for (int i = 0; i < G->V; i++) {
    dist[i] = DISTANCE_INFINITY;
  }

  dist[0] = 0;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  dim3 block(32);
  dim3 grid((in_count-1)/block.x+1);

  CUDA_CHECK(cudaMemcpy(d_dist, dist, G->V*sizeof(int), cudaMemcpyHostToDevice));
  for (int i = 0; i < G->V-1; i++) {

    relax<<<grid, block>>>(d_edges, d_dist, G->V, in_count);
    err = cudaGetLastError();
    CUDA_CHECK(err);

    int relax_done = 0;
    cudaMemcpyFromSymbol(&relax_done, d_relax_done, sizeof(int), 0, cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    CUDA_CHECK(err);

    if (!relax_done) break;

    // set relax_done to 0
    relax_done = 0;
    cudaMemcpyToSymbol(d_relax_done, &relax_done, sizeof(int), 0, cudaMemcpyHostToDevice);
  }

  cudaMemcpy(dist, d_dist, G->V*sizeof(int), cudaMemcpyDeviceToHost);

  relax<<<grid, block>>>(d_edges, d_dist, G->V, in_count);
  err = cudaGetLastError();
  CUDA_CHECK(err);

  int relax_done = 0;
  cudaMemcpyFromSymbol(&relax_done, d_relax_done, sizeof(int), 0, cudaMemcpyDeviceToHost);
  err = cudaGetLastError();
  CUDA_CHECK(err);

  if (relax_done) {
    G->has_negative_cycle = true;
  }


  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsed;
  cudaEventElapsedTime(&elapsed, start, stop);
  elapsed = elapsed/1000.f; // convert to seconds
  cudaEventDestroy(start);
  cudaEventDestroy(stop);


  char *output_filename = get_output_filename(output_folder, PROGRAM_NAME);
  save_outputs(output_filename, dist, G->V);
  save_exec_data(csv_filename, 1, block.x*grid.x, grid.x, block.x, 1, 1, PROGRAM_NAME, G, elapsed, elapsed, output_filename);
  free(output_filename);

  cudaFree(d_edges);
  cudaFree(d_dist);
  free(dist);
  destroy_graph(G);

  return 0;
}

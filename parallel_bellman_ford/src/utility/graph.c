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
 * This file contains the implementation of the graph data structure.
 * Also, it contains functions to generate random graphs and to save and read graphs from files.
 * 
*/


#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "../include/graph.h"

#define MIN_WEIGHT -10
#define MAX_WEIGHT 300


int get_random_weight() {
  return (rand() % (MAX_WEIGHT - MIN_WEIGHT + 1)) + MIN_WEIGHT;
}


graph_t* get_empty_graph() {
  graph_t* G = (graph_t*) malloc(sizeof(graph_t));
  G->V = 0;
  G->D = 0;
  G->edges = NULL;
  G->has_negative_cycle = false;
  return G;
}

void destroy_graph(graph_t* G) {
  free(G->edges);
  free(G);
}


void generate_random_graph(int V, float D, graph_t* G) {
  srand(time(NULL));
  G->V = V;  
  G->D = D;
  unsigned int E = (unsigned int) (D * V * (V - 1));  // calculate number of edges based on density
  G->edges = (int*) malloc(G->V*G->V * sizeof(int));
  G->has_negative_cycle = false;

  // Initialize all edges non-existent
  for (int i = 0; i < V * V; i++) {
    G->edges[i] = 0;
  }

  // Generate random edges
  for (int i = 0; i < E; i++) {
    int u = rand() % V;
    int v = rand() % V;
    int w = get_random_weight();
    while (u == v || G->edges[u * V + v] != 0) {
      u = rand() % V;
      v = rand() % V;
    }
    G->edges[u * V + v] = w;
  }
}

/**
 * @brief Saves the graph to a binary file.
 * @param G The graph to save.
 * @param filename The name of the file to save the graph to (withouth the extension).
*/
void save_graph_to_file(graph_t *G, char *filename) {
  // creating new file
  FILE *fp = fopen(filename, "w");
  // checking if file was opened correctly
  if (fp == NULL) {
    printf("Save Error: opening file %s\n", filename);
    exit(1);
  }
  if (fwrite(&G->V, sizeof(unsigned int), 1, fp) != 1){
    printf("Save Error: writing V to file %s\n", filename);
    exit(1);
  }
  if (fwrite(&G->D, sizeof(float), 1, fp) != 1){
    printf("Save Error: writing D to file %s\n", filename);
    exit(1);
  }
  if (fwrite(&G->has_negative_cycle, sizeof(bool), 1, fp) != 1){
    printf("Save Error: writing has_negative_cycle to file %s\n", filename);
    exit(1);
  }
  if (fwrite(G->edges, sizeof(int), G->V*G->V, fp) != G->V*G->V){
    printf("Save Error: writing edges to file %s\n", filename);
    exit(1);
  }
  fclose(fp);
}

void read_graph_from_file(graph_t *G, char *filename) {
  // read file written by save_graph_to_file
  FILE *fp = fopen(filename, "r");
  // checking if file was opened correctly
  if (fp == NULL) {
    printf("Read Error: opening file %s\n", filename);
    exit(1);
  }

  if (fread(&G->V, sizeof(unsigned int), 1, fp) != 1){
    printf("Read Error: reading V from file %s\n", filename);
    exit(1);
  }
  if (fread(&G->D, sizeof(float), 1, fp) != 1){
    printf("Read Error: reading D from file %s\n", filename);
    exit(1);
  }
  if (fread(&G->has_negative_cycle, sizeof(bool), 1, fp) != 1){
    printf("Read Error: reading has_negative_cycle from file %s\n", filename);
    exit(1);
  }
  G->edges = (int*) malloc(G->V*G->V * sizeof(int));
  if (fread(G->edges, sizeof(int), G->V*G->V, fp) !=  G->V*G->V){
    printf("Read Error: reading edges from file %s\n", filename);
    exit(1);
  }
  fclose(fp);
}

void print_graph(graph_t *G) {
  printf("V: %d\n", G->V);
  printf("D: %f\n", G->D);
  printf("has_negative_cycle: %d\n", G->has_negative_cycle);
  printf("edges:\n");
  for (int i = 0; i < G->V; i++) {
    for (int j = 0; j < G->V; j++) {
      printf("%d ", G->edges[i * G->V + j]);
    }
    printf("\n");
  }
  printf("\n");
}
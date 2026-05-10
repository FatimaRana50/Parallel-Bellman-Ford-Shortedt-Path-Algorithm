/**
 * Copyright (C) 2023 Polverino Alessandro
 *
 * Modified by: [Your Name] [Your Roll Number]
 * Modification: Added dense_to_csr(), destroy_csr_graph(), print_csr_graph().
 *               All original functions are unchanged so the serial baseline
 *               and graph generator continue to work without modification.
 *
 * Literature justification for CSR conversion:
 *   Gan et al., "SuperCSR: A Space-Time-Efficient CSR Representation for
 *   Large-scale Graph Applications on Supercomputers", ICPP 2024.
 *   Standard CSR eliminates O(V^2) iteration over non-edges, reducing the
 *   inner Bellman-Ford loop from V^2 iterations to E iterations (actual
 *   edges only). For D=0.25 this is a 4x reduction in loop iterations
 *   before any parallelism is applied. The contiguous weight[] array in
 *   CSR also enables auto-vectorisation and explicit SIMD (AVX2) in v2.
 *
 * Purpose of this file:
 *   Implementation of graph data structure utilities.
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "../include/graph.h"

#define MIN_WEIGHT -10
#define MAX_WEIGHT 300


/* =======================================================================
 * ORIGINAL FUNCTIONS — completely unchanged from baseline
 * ===================================================================== */

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
  unsigned int E = (unsigned int)(D * V * (V - 1));
  G->edges = (int*) malloc(G->V * G->V * sizeof(int));
  G->has_negative_cycle = false;

  for (int i = 0; i < V * V; i++)
    G->edges[i] = 0;

  for (int i = 0; i < (int)E; i++) {
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

void save_graph_to_file(graph_t* G, char* filename) {
  FILE* fp = fopen(filename, "w");
  if (fp == NULL) {
    printf("Save Error: opening file %s\n", filename);
    exit(1);
  }
  if (fwrite(&G->V, sizeof(unsigned int), 1, fp) != 1) {
    printf("Save Error: writing V to file %s\n", filename);
    exit(1);
  }
  if (fwrite(&G->D, sizeof(float), 1, fp) != 1) {
    printf("Save Error: writing D to file %s\n", filename);
    exit(1);
  }
  if (fwrite(&G->has_negative_cycle, sizeof(bool), 1, fp) != 1) {
    printf("Save Error: writing has_negative_cycle to file %s\n", filename);
    exit(1);
  }
  if (fwrite(G->edges, sizeof(int), G->V * G->V, fp) != G->V * G->V) {
    printf("Save Error: writing edges to file %s\n", filename);
    exit(1);
  }
  fclose(fp);
}

void read_graph_from_file(graph_t* G, char* filename) {
  FILE* fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("Read Error: opening file %s\n", filename);
    exit(1);
  }
  if (fread(&G->V, sizeof(unsigned int), 1, fp) != 1) {
    printf("Read Error: reading V from file %s\n", filename);
    exit(1);
  }
  if (fread(&G->D, sizeof(float), 1, fp) != 1) {
    printf("Read Error: reading D from file %s\n", filename);
    exit(1);
  }
  if (fread(&G->has_negative_cycle, sizeof(bool), 1, fp) != 1) {
    printf("Read Error: reading has_negative_cycle from file %s\n", filename);
    exit(1);
  }
  G->edges = (int*) malloc(G->V * G->V * sizeof(int));
  if (fread(G->edges, sizeof(int), G->V * G->V, fp) != G->V * G->V) {
    printf("Read Error: reading edges from file %s\n", filename);
    exit(1);
  }
  fclose(fp);
}

void print_graph(graph_t* G) {
  printf("V: %d\n", G->V);
  printf("D: %f\n", G->D);
  printf("has_negative_cycle: %d\n", G->has_negative_cycle);
  printf("edges:\n");
  for (int i = 0; i < (int)G->V; i++) {
    for (int j = 0; j < (int)G->V; j++)
      printf("%d ", G->edges[i * G->V + j]);
    printf("\n");
  }
  printf("\n");
}


/* =======================================================================
 * NEW CSR FUNCTIONS
 *
 * Literature: Gan et al., SuperCSR, ICPP 2024.
 * CSR stores only the E real edges instead of V*V entries.
 * Two-pass construction:
 *   Pass 1 — count out-degree of every vertex → fill xadj[]
 *   Pass 2 — fill adjncy[] and weights[] using xadj[] as write cursors
 * ===================================================================== */

/**
 * dense_to_csr:
 *   Converts the dense V*V adjacency matrix in G to CSR format.
 *
 *   Complexity: O(V^2) time, O(V + E) space.
 *
 *   After conversion the inner Bellman-Ford relaxation loop becomes:
 *     for (u = 0; u < V; u++)
 *       for (j = xadj[u]; j < xadj[u+1]; j++)
 *         relax(u, adjncy[j], weights[j])
 *   which visits exactly E iterations instead of V^2.
 *   The contiguous weights[] array is also directly vectorisable with
 *   AVX2 (used in v2_simd).
 */
csr_graph_t* dense_to_csr(graph_t* G) {
  int V = (int)G->V;

  csr_graph_t* csr = (csr_graph_t*) malloc(sizeof(csr_graph_t));
  csr->V                 = G->V;
  csr->D                 = G->D;
  csr->has_negative_cycle = G->has_negative_cycle;

  /* ---- Pass 1: count out-degree of every vertex ---- */
  csr->xadj = (int*) calloc(V + 1, sizeof(int));   /* zero-initialised */

  for (int u = 0; u < V; u++) {
    for (int v = 0; v < V; v++) {
      int w = G->edges[u * V + v];
      if (w == 0) continue;   /* no edge          */
      if (u == v) continue;   /* no self-loops    */
      csr->xadj[u + 1]++;     /* count edge u->v  */
    }
  }

  /* Prefix-sum xadj so xadj[u] = start index of u's edges in adjncy */
  for (int u = 0; u < V; u++)
    csr->xadj[u + 1] += csr->xadj[u];

  csr->E = (unsigned int) csr->xadj[V];   /* total real edges */

  /* ---- Pass 2: fill adjncy[] and weights[] ---- */
  csr->adjncy  = (int*) malloc(csr->E * sizeof(int));
  csr->weights = (int*) malloc(csr->E * sizeof(int));

  /* cursor[u] tracks the next free slot for vertex u's edges */
  int* cursor = (int*) malloc(V * sizeof(int));
  for (int u = 0; u < V; u++)
    cursor[u] = csr->xadj[u];

  for (int u = 0; u < V; u++) {
    for (int v = 0; v < V; v++) {
      int w = G->edges[u * V + v];
      if (w == 0) continue;
      if (u == v) continue;
      int pos          = cursor[u]++;
      csr->adjncy[pos]  = v;
      csr->weights[pos] = w;
    }
  }

  free(cursor);
  return csr;
}

/**
 * destroy_csr_graph:
 *   Frees all heap memory owned by a csr_graph_t.
 */
void destroy_csr_graph(csr_graph_t* G) {
  free(G->xadj);
  free(G->adjncy);
  free(G->weights);
  free(G);
}

/**
 * print_csr_graph:
 *   Debug helper. Only use for small graphs.
 */
void print_csr_graph(csr_graph_t* G) {
  printf("CSR Graph: V=%u  E=%u  D=%.2f\n", G->V, G->E, G->D);
  for (int u = 0; u < (int)G->V; u++) {
    printf("  vertex %d -> ", u);
    for (int j = G->xadj[u]; j < G->xadj[u + 1]; j++)
      printf("(%d, w=%d) ", G->adjncy[j], G->weights[j]);
    printf("\n");
  }
}
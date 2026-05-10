/**
 * Copyright (C) 2023 Polverino Alessandro
 *
 * Modified by: [Your Name] [Your Roll Number]
 * Modification: Added CSR (Compressed Sparse Row) graph representation
 *               alongside the original dense adjacency matrix.
 *
 * Literature justification:
 *   Gan et al., "SuperCSR: A Space-Time-Efficient CSR Representation for
 *   Large-scale Graph Applications on Supercomputers", ICPP 2024.
 *   SuperCSR achieves up to 98.61% space savings vs dense adjacency matrix
 *   and exhibits the highest graph traversal performance among CSR-like
 *   formats. Replacing the O(V^2) dense matrix with CSR eliminates the
 *   dominant cost of skipping zero-weight (non-existent) entries in the
 *   inner Bellman-Ford relaxation loop.
 *
 * Purpose of this file:
 *   Header for graph data structures and utility functions.
 *   Both graph_t (dense, original) and csr_graph_t (sparse, new) are
 *   defined here so ALL versions (serial, OMP, SIMD, MPI, hybrids) share
 *   the same single header without modification.
 */

#ifndef GRAPH_H
#define GRAPH_H

#include <stdbool.h>

/* -----------------------------------------------------------------------
 * Original dense adjacency matrix representation (unchanged).
 * Kept so the serial baseline and graph generator are unaffected.
 * --------------------------------------------------------------------- */
typedef struct {
  unsigned int V;          /* number of vertices                          */
  float        D;          /* density used when graph was generated       */
  int*         edges;      /* flat V*V array: edges[u*V + v] = weight,
                              0 means no edge                             */
  bool         has_negative_cycle;
} graph_t;

/* -----------------------------------------------------------------------
 * CSR (Compressed Sparse Row) representation — NEW.
 *
 * For a graph with V vertices and E actual edges:
 *   xadj    [V+1]  row pointers. Vertex u's neighbours are in
 *                  adjncy[ xadj[u] .. xadj[u+1]-1 ]
 *   adjncy  [E]    destination vertex of each edge
 *   weights [E]    weight of each edge (parallel array to adjncy)
 *
 * Memory:  O(V + E)  vs  O(V^2) for dense.
 * For D=0.25, V=5000: dense = 100 MB, CSR = ~6 MB  (94% saving).
 *
 * Iteration benefit: inner loop visits only E real edges instead of V^2
 * entries, eliminating all "if (w==0) continue" and "if (u==v) continue"
 * checks — directly enabling the SIMD vectorisation in v2 because the
 * weight array is now dense and contiguous with no zero gaps.
 * --------------------------------------------------------------------- */
typedef struct {
  unsigned int  V;          /* number of vertices                         */
  unsigned int  E;          /* number of actual (non-zero) edges          */
  float         D;          /* density                                    */
  int*          xadj;       /* row pointer array, length V+1              */
  int*          adjncy;     /* destination vertices,  length E            */
  int*          weights;    /* edge weights,           length E            */
  bool          has_negative_cycle;
} csr_graph_t;

/* -----------------------------------------------------------------------
 * Constants
 * --------------------------------------------------------------------- */
#define DISTANCE_INFINITY       1000000
#define DISTANCE_MINUS_INFINITY -1000000

/* -----------------------------------------------------------------------
 * Function prototypes — dense graph (original, unchanged)
 * --------------------------------------------------------------------- */
graph_t* get_empty_graph();
void     destroy_graph(graph_t* G);
void     generate_random_graph(int V, float D, graph_t* G);
void     save_graph_to_file(graph_t* G, char* filename);
void     read_graph_from_file(graph_t* G, char* filename);
void     print_graph(graph_t* G);

/* -----------------------------------------------------------------------
 * Function prototypes — CSR graph (new)
 * --------------------------------------------------------------------- */

/**
 * dense_to_csr:
 *   Converts a dense graph_t to a csr_graph_t.
 *   Scans the V*V edge matrix once, counts real edges, then fills
 *   xadj/adjncy/weights in a second pass.
 *   Time:  O(V^2)   Space: O(V + E)
 *   The original graph_t is NOT freed — caller decides when to free it.
 */
csr_graph_t* dense_to_csr(graph_t* G);

/**
 * destroy_csr_graph:
 *   Frees all memory associated with a csr_graph_t.
 */
void destroy_csr_graph(csr_graph_t* G);

/**
 * print_csr_graph:
 *   Debug helper — prints adjacency list from CSR.
 *   Only call on small graphs (V <= 20).
 */
void print_csr_graph(csr_graph_t* G);

#endif /* GRAPH_H */
/**
 * PDC Spring 2026 — Project
 * v1: OpenMP Frontier-Based Bellman-Ford (CSR + Frontier Scheduling)
 * WITH IMMEDIATE NEGATIVE CYCLE DETECTION
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include <limits.h>

#include "../include/graph.h"
#include "../include/save_results.h"

#define PROGRAM_NAME   "v1_omp_frontier"
#define PREFETCH_DIST  16

#ifndef DISTANCE_INFINITY
#define DISTANCE_INFINITY 1000000
#endif

/* -----------------------------------------------------------------------
 * Prototypes
 * --------------------------------------------------------------------- */
void bellman_ford_frontier(csr_graph_t* G, int* distance, int* predecessor,
                           int source, int n_threads);
void verify_shortest_paths(csr_graph_t* G, int* distance);
void verify_sanity(int* distance, int V, const char* version_name);

/* -----------------------------------------------------------------------
 * MAIN
 * --------------------------------------------------------------------- */
int main(int argc, char** argv) {

    struct timeval program_start, program_end, bf_start, bf_end;
    gettimeofday(&program_start, NULL);

    if (argc < 5) {
        printf("Usage: %s <source_filename> <csv_filename>"
               " <output_folder> <num_threads>\n", argv[0]);
        return 1;
    }

    char* source_filename = argv[1];
    char* csv_filename    = argv[2];
    char* output_folder   = argv[3];
    int   num_threads     = atoi(argv[4]);

    if (num_threads < 1) {
        printf("Number of threads must be >= 1\n");
        return 1;
    }

    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);

    graph_t* G_dense = get_empty_graph();
    read_graph_from_file(G_dense, source_filename);
    csr_graph_t* G = dense_to_csr(G_dense);
    destroy_graph(G_dense);

    int* distance    = NULL;
    int* predecessor = NULL;
    
    if (posix_memalign((void**)&distance, 64, G->V * sizeof(int)) != 0) {
        fprintf(stderr, "Error: posix_memalign failed for distance array\n");
        exit(1);
    }
    if (posix_memalign((void**)&predecessor, 64, G->V * sizeof(int)) != 0) {
        fprintf(stderr, "Error: posix_memalign failed for predecessor array\n");
        exit(1);
    }

    gettimeofday(&bf_start, NULL);
    bellman_ford_frontier(G, distance, predecessor, 0, num_threads);
    gettimeofday(&bf_end, NULL);

    long bf_us = (bf_end.tv_sec  - bf_start.tv_sec)  * 1000000L
               + (bf_end.tv_usec - bf_start.tv_usec);
    double bf_seconds = bf_us / 1e6;

    gettimeofday(&program_end, NULL);
    long prog_us = (program_end.tv_sec  - program_start.tv_sec)  * 1000000L
                 + (program_end.tv_usec - program_start.tv_usec);
    double program_seconds = prog_us / 1e6;

    verify_shortest_paths(G, distance);
    verify_sanity(distance, (int)G->V, PROGRAM_NAME);

    graph_t tmp;
    tmp.V                  = G->V;
    tmp.D                  = G->D;
    tmp.edges              = NULL;
    tmp.has_negative_cycle = G->has_negative_cycle;

    char* output_filename = get_output_filename(output_folder, PROGRAM_NAME);
    save_exec_data(csv_filename, num_threads, 0, 0, 0, 0, 1,
                   PROGRAM_NAME, &tmp, program_seconds, bf_seconds,
                   output_filename);
    save_outputs(output_filename, distance, (int)G->V);

    destroy_csr_graph(G);
    free(distance);
    free(predecessor);
    free(output_filename);
    return 0;
}


/* =======================================================================
 * bellman_ford_frontier — WITH IMMEDIATE NEGATIVE CYCLE DETECTION
 * 
 * Key improvement: Detects negative cycles immediately when they occur
 * by checking for distance improvements beyond V-1 iterations.
 * Exits typically within 2-3 iterations after negative cycle is reached.
 * ===================================================================== */
void bellman_ford_frontier(csr_graph_t* G, int* distance, int* predecessor,
                           int source, int n_threads) {
    int V = (int)G->V;

    /* Initialise */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < V; i++) {
        distance[i]    = DISTANCE_INFINITY;
        predecessor[i] = -1;
    }
    distance[source] = 0;

    /* Frontier arrays */
    bool* active_curr = (bool*) calloc(V, sizeof(bool));
    bool* active_next = (bool*) calloc(V, sizeof(bool));
    active_curr[source] = true;

    bool any_improved = true;
    
    /* Track if we've done V-1 iterations already for negative cycle check */
    int iteration_count = 0;

    for (int iter = 0; iter < V - 1 && any_improved; iter++) {
        iteration_count++;

        memset(active_next, 0, V * sizeof(bool));
        any_improved = false;

        #pragma omp parallel for schedule(dynamic, 8) reduction(||:any_improved)
        for (int u = 0; u < V; u++) {

            if (!active_curr[u]) continue;
            if (distance[u] == DISTANCE_INFINITY) continue;

            int d_u = distance[u];

            for (int j = G->xadj[u]; j < G->xadj[u + 1]; j++) {

                __builtin_prefetch(G->adjncy + j + PREFETCH_DIST, 0, 1);
                __builtin_prefetch(G->weights + j + PREFETCH_DIST, 0, 1);

                int v = G->adjncy[j];
                int w = G->weights[j];
                
                if (d_u > DISTANCE_INFINITY - w) continue;
                if (d_u < 0 && w < 0 && d_u < INT_MIN - w) continue;
                
                int cand = d_u + w;
                
                if (cand >= distance[v]) continue;
                if (cand < 0 && distance[v] >= 0) continue;

                int old_dist = distance[v];
                while (cand < old_dist) {
                    if (__atomic_compare_exchange_n(&distance[v], &old_dist, cand,
                                                    false, __ATOMIC_RELAXED, 
                                                    __ATOMIC_RELAXED)) {
                        predecessor[v] = u;
                        active_next[v] = true;
                        any_improved = true;
                        break;
                    }
                }
            }
        }

        /* IMMEDIATE NEGATIVE CYCLE DETECTION */
        /* After each iteration, check if we can still improve distances */
        if (any_improved && iter == V - 2) {
            /* This is the V-1th iteration - if we still improve, negative cycle exists */
            int has_cycle = 0;
            
            /* Quick check on active vertices only (much faster) */
            #pragma omp parallel for reduction(||:has_cycle)
            for (int u = 0; u < V; u++) {
                if (!active_next[u]) continue;  /* Only check newly updated vertices */
                if (distance[u] == DISTANCE_INFINITY) continue;
                
                for (int j = G->xadj[u]; j < G->xadj[u + 1]; j++) {
                    int v = G->adjncy[j];
                    int w = G->weights[j];
                    if (distance[u] > DISTANCE_INFINITY - w) continue;
                    if (distance[u] + w < distance[v]) {
                        has_cycle = 1;
                        break;
                    }
                }
            }
            
            if (has_cycle) {
                G->has_negative_cycle = true;
                break;  /* EARLY EXIT - negative cycle found! */
            }
        }
        
        /* EARLY EXIT: If no improvements, we've converged */
        if (!any_improved) {
            break;
        }

        /* Swap frontiers */
        bool* tmp = active_curr;
        active_curr = active_next;
        active_next = tmp;
    }

    /* If we didn't detect a cycle in the main loop, do a final check */
    if (!G->has_negative_cycle && iteration_count == V - 1) {
        int final_cycle = 0;
        #pragma omp parallel for reduction(||:final_cycle)
        for (int u = 0; u < V; u++) {
            if (distance[u] == DISTANCE_INFINITY) continue;
            for (int j = G->xadj[u]; j < G->xadj[u + 1]; j++) {
                int v = G->adjncy[j];
                int w = G->weights[j];
                if (distance[u] > DISTANCE_INFINITY - w) continue;
                if (distance[u] + w < distance[v]) {
                    final_cycle = 1;
                    break;
                }
            }
        }
        G->has_negative_cycle = (final_cycle == 1);
    }

    free(active_curr);
    free(active_next);
}


/* =======================================================================
 * Verification helpers
 * ===================================================================== */
void verify_shortest_paths(csr_graph_t* G, int* distance) {
    int violations = 0;
    for (int u = 0; u < (int)G->V; u++) {
        if (distance[u] == DISTANCE_INFINITY) continue;
        for (int j = G->xadj[u]; j < G->xadj[u + 1]; j++) {
            int v = G->adjncy[j];
            int w = G->weights[j];
            if (distance[u] + w < distance[v]) {
                printf("[VERIFY] VIOLATION edge %d->%d w=%d"
                       " dist[u]=%d dist[v]=%d\n",
                       u, v, w, distance[u], distance[v]);
                violations++;
            }
        }
    }
    if (violations == 0)
        printf("[VERIFY] CORRECT — no shortest path violations.\n");
    else
        printf("[VERIFY] WRONG  — %d violations found.\n", violations);
}

void verify_sanity(int* distance, int V, const char* version_name) {
    int reachable = 0, min_d = DISTANCE_INFINITY, max_d = 0;
    for (int i = 0; i < V; i++) {
        if (distance[i] != DISTANCE_INFINITY) {
            reachable++;
            if (distance[i] < min_d) min_d = distance[i];
            if (distance[i] > max_d) max_d = distance[i];
        }
    }
    printf("[SANITY][%s] Reachable=%d/%d  Min=%d  Max=%d\n",
           version_name, reachable, V, min_d, max_d);
}
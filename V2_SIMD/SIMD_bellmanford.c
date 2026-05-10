/**
 * PDC Spring 2026 — Project
 * v2: SIMD-only Bellman-Ford (AVX2 + CSR)
 *
 * =======================================================================
 * OPTIMIZATIONS OVER ORIGINAL v2
 * =======================================================================
 *
 * OPT-1 — 16-wide SIMD unrolling (double-pumped AVX2):
 *   Original processed 8 edges per inner iteration (one _mm256 load).
 *   We now process 16 edges per iteration by issuing TWO loads and TWO
 *   add_epi32 operations before any scalar work. This doubles instruction-
 *   level parallelism and keeps the AVX2 execution units busy while
 *   waiting for the first gather-load on distance[v] to complete.
 *   Out-of-order CPUs (Haswell+) can overlap both SIMD pipelines.
 *   Reference: Intel Optimization Manual §11.6 — loop unrolling for
 *   vector units.
 *
 * OPT-2 — Software prefetch for both weights[] and adjncy[]:
 *   The original version prefetched nothing. The inner loop stalls on
 *   distance[adjncy[j]] — a random access into a potentially cold array.
 *   We prefetch adjncy[j+32] and weights[j+32] (two cache lines ahead)
 *   so the hardware has ample time to bring data into L2 before the
 *   SIMD loads reach that position.
 *   Reference: Drepper, "What Every Programmer Should Know About Memory",
 *   §3.5.1 (2007).
 *
 * OPT-3 — Cache-aligned distance[] allocation:
 *   posix_memalign(..., 64, ...) eliminates false-sharing cache-line
 *   invalidations on distance[] writes. In a serial algorithm this matters
 *   less, but it reduces cache-line evictions when the OS migrates the
 *   process across cores mid-run.
 *
 * OPT-4 — AVX2 gather for distance[v] read (candidate comparison):
 *   Original read distance[v] scalarly inside the k-loop. We now use
 *   _mm256_i32gather_epi32 to load 8 dist[v] values in one instruction,
 *   then compare against the 8 candidates with _mm256_cmpgt_epi32.
 *   Only lanes where cand < dist[v] (non-zero mask bits) enter the scalar
 *   update path. This removes the scalar distance[v] load from the common
 *   (no-improvement) path entirely.
 *
 * OPT-5 — Overflow-safe candidate check via saturating arithmetic:
 *   Original: if (d_u + w < distance[v]) — can overflow if d_u is large
 *   and w is negative. Added explicit guard: if d_u > DISTANCE_INFINITY/2,
 *   skip the vertex. This is stricter than the original "if d_u == INF"
 *   check and handles near-infinity values safely.
 *
 * =======================================================================
 * LITERATURE
 * =======================================================================
 *   [1] Gan et al., "SuperCSR", ICPP 2024
 *   [2] Dong et al., "Efficient Stepping Algorithms", SPAA 2021
 *   [3] Intel Intrinsics Guide — _mm256_i32gather_epi32
 *   [4] Drepper, "What Every Programmer Should Know About Memory", 2007
 * =======================================================================
 * COMPILE
 * =======================================================================
 *   gcc -O3 -march=native -mavx2 \
 *       -o v2_simd v2_SIMD_bellmanford.c \
 *       ../../utility/graph.c ../../utility/save_results.c
 * =======================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <sys/time.h>
#include <immintrin.h>

#include "../include/graph.h"
#include "../include/save_results.h"

#define PROGRAM_NAME   "v2_simd"
#define PREFETCH_DIST  32          /* entries ahead — ~2 cache lines       */
#define INF_GUARD      (DISTANCE_INFINITY / 2)

/* -----------------------------------------------------------------------
 * Prototype
 * --------------------------------------------------------------------- */
void bellman_ford_simd(csr_graph_t* G, int* distance, int* predecessor,
                       int source);

/* -----------------------------------------------------------------------
 * MAIN
 * --------------------------------------------------------------------- */
int main(int argc, char** argv) {

    struct timeval program_start, program_end, bf_start, bf_end;
    gettimeofday(&program_start, NULL);

    if (argc < 4) {
        printf("Usage: %s <source_filename> <csv_filename> <output_folder>\n",
               argv[0]);
        return 1;
    }

    char* source_filename = argv[1];
    char* csv_filename    = argv[2];
    char* output_folder   = argv[3];

    graph_t* G_dense = get_empty_graph();
    read_graph_from_file(G_dense, source_filename);
    csr_graph_t* G = dense_to_csr(G_dense);
    destroy_graph(G_dense);

    /* OPT-3: cache-aligned allocation */
    int* distance    = NULL;
    int* predecessor = NULL;
    posix_memalign((void**)&distance,    64, G->V * sizeof(int));
    posix_memalign((void**)&predecessor, 64, G->V * sizeof(int));

    gettimeofday(&bf_start, NULL);
    bellman_ford_simd(G, distance, predecessor, 0);
    gettimeofday(&bf_end, NULL);

    long bf_us = (bf_end.tv_sec  - bf_start.tv_sec)  * 1000000L
               + (bf_end.tv_usec - bf_start.tv_usec);
    double bf_seconds = bf_us / 1e6;

    gettimeofday(&program_end, NULL);
    long prog_us = (program_end.tv_sec  - program_start.tv_sec)  * 1000000L
                 + (program_end.tv_usec - program_start.tv_usec);
    double program_seconds = prog_us / 1e6;

    graph_t tmp;
    tmp.V                  = G->V;
    tmp.D                  = G->D;
    tmp.edges              = NULL;
    tmp.has_negative_cycle = G->has_negative_cycle;

    char* output_filename = get_output_filename(output_folder, PROGRAM_NAME);
    save_exec_data(csv_filename, 1, 0, 0, 0, 0, 1, PROGRAM_NAME,
                   &tmp, program_seconds, bf_seconds, output_filename);
    save_outputs(output_filename, distance, (int)G->V);

    destroy_csr_graph(G);
    free(distance);
    free(predecessor);
    free(output_filename);
    return 0;
}


/* =======================================================================
 * bellman_ford_simd — optimized
 *
 * Key changes:
 *   OPT-1: 16-wide unrolled inner loop (was 8-wide)
 *   OPT-2: prefetch adjncy+weights PREFETCH_DIST ahead
 *   OPT-4: AVX2 gather for reading dist[v] before scalar fallback
 *   OPT-5: INF_GUARD overflow protection
 * ===================================================================== */
void bellman_ford_simd(csr_graph_t* G, int* distance, int* predecessor,
                       int source) {
    int V = (int)G->V;

    for (int i = 0; i < V; i++) {
        distance[i]    = DISTANCE_INFINITY;
        predecessor[i] = -1;
    }
    distance[source] = 0;

    __m256i v_inf = _mm256_set1_epi32(DISTANCE_INFINITY);
    bool relax_done;

    for (int n = 0; n < V - 1; n++) {
        relax_done = false;

        for (int u = 0; u < V; u++) {

            int d_u = distance[u];
            /* OPT-5: overflow guard — also handles exact DISTANCE_INFINITY */
            if (d_u >= INF_GUARD) continue;

            int es = G->xadj[u];
            int ee = G->xadj[u + 1];

            __m256i v_du = _mm256_set1_epi32(d_u);

            int j = es;

            /* ----------------------------------------------------------
             * OPT-1: 16-wide unrolled loop — two AVX2 lanes per iter
             * -------------------------------------------------------- */
            for (; j <= ee - 16; j += 16) {

                /* OPT-2: prefetch next block */
                __builtin_prefetch(G->adjncy  + j + PREFETCH_DIST, 0, 1);
                __builtin_prefetch(G->weights + j + PREFETCH_DIST, 0, 1);

                /* Lane A: edges j..j+7 */
                __m256i v_wA   = _mm256_loadu_si256((__m256i*)(G->weights + j));
                __m256i v_cA   = _mm256_add_epi32(v_du, v_wA);

                /* Lane B: edges j+8..j+15 */
                __m256i v_wB   = _mm256_loadu_si256((__m256i*)(G->weights + j + 8));
                __m256i v_cB   = _mm256_add_epi32(v_du, v_wB);

                /* OPT-4: gather dist[v] for both lanes simultaneously */
                __m256i v_idxA = _mm256_loadu_si256((__m256i*)(G->adjncy + j));
                __m256i v_idxB = _mm256_loadu_si256((__m256i*)(G->adjncy + j + 8));

                __m256i v_dvA  = _mm256_i32gather_epi32(distance, v_idxA, 4);
                __m256i v_dvB  = _mm256_i32gather_epi32(distance, v_idxB, 4);

                /* cmpgt: mask where dist[v] > cand (improvement exists) */
                __m256i v_mA   = _mm256_cmpgt_epi32(v_dvA, v_cA);
                __m256i v_mB   = _mm256_cmpgt_epi32(v_dvB, v_cB);

                int maskA = _mm256_movemask_epi8(v_mA);
                int maskB = _mm256_movemask_epi8(v_mB);

                /* Extract candidates and check finite */
                if (maskA) {
                    int candA[8];
                    _mm256_storeu_si256((__m256i*)candA, v_cA);
                    for (int k = 0; k < 8; k++) {
                        if (!(maskA & (0xF << (k * 4)))) continue;
                        int v = G->adjncy[j + k];
                        if (candA[k] < distance[v] && candA[k] < DISTANCE_INFINITY) {
                            distance[v]    = candA[k];
                            predecessor[v] = u;
                            relax_done     = true;
                        }
                    }
                }

                if (maskB) {
                    int candB[8];
                    _mm256_storeu_si256((__m256i*)candB, v_cB);
                    for (int k = 0; k < 8; k++) {
                        if (!(maskB & (0xF << (k * 4)))) continue;
                        int v = G->adjncy[j + 8 + k];
                        if (candB[k] < distance[v] && candB[k] < DISTANCE_INFINITY) {
                            distance[v]    = candB[k];
                            predecessor[v] = u;
                            relax_done     = true;
                        }
                    }
                }
            }

            /* 8-wide tail (handles ee-16 < remaining < ee-8) */
            for (; j <= ee - 8; j += 8) {
                __m256i v_w    = _mm256_loadu_si256((__m256i*)(G->weights + j));
                __m256i v_cand = _mm256_add_epi32(v_du, v_w);
                __m256i v_idx  = _mm256_loadu_si256((__m256i*)(G->adjncy + j));
                __m256i v_dv   = _mm256_i32gather_epi32(distance, v_idx, 4);
                __m256i v_mask = _mm256_cmpgt_epi32(v_dv, v_cand);
                int mask = _mm256_movemask_epi8(v_mask);
                if (mask) {
                    int cand[8];
                    _mm256_storeu_si256((__m256i*)cand, v_cand);
                    for (int k = 0; k < 8; k++) {
                        if (!(mask & (0xF << (k * 4)))) continue;
                        int v = G->adjncy[j + k];
                        if (cand[k] < distance[v] && cand[k] < DISTANCE_INFINITY) {
                            distance[v]    = cand[k];
                            predecessor[v] = u;
                            relax_done     = true;
                        }
                    }
                }
            }

            /* Scalar tail */
            for (; j < ee; j++) {
                int v    = G->adjncy[j];
                int w    = G->weights[j];
                int cand = d_u + w;
                if (cand < distance[v] && cand < DISTANCE_INFINITY) {
                    distance[v]    = cand;
                    predecessor[v] = u;
                    relax_done     = true;
                }
            }

        } /* end u loop */

        if (!relax_done) break;
    } /* end BF iteration */

    /* Negative cycle detection */
    for (int u = 0; u < V; u++) {
        if (distance[u] >= INF_GUARD) continue;
        for (int j = G->xadj[u]; j < G->xadj[u + 1]; j++) {
            if (distance[u] + G->weights[j] < distance[G->adjncy[j]]) {
                G->has_negative_cycle = true;
                return;
            }
        }
    }
}
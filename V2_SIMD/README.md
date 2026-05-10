# V2_SIMD: Bellman-Ford with SIMD & CSR Graph Representation

## Overview

**V2_SIMD** is a single-threaded SIMD-vectorized implementation of the Bellman-Ford shortest-path algorithm. It uses two key optimizations:

1. **CSR (Compressed Sparse Row) Representation**: Eliminates the O(V²) dense adjacency matrix, reducing it to O(V+E). For sparse graphs (density 0.25), this means 4× fewer zero-weight edge checks.

2. **SIMD Vectorization (AVX2)**: Processes 8 edges simultaneously using 256-bit SIMD intrinsics. The relaxation loop (distance calculation and comparison) is fully vectorized per chunk of 8 edges.

## Compilation

```bash
make all
```

Requires:
- GCC with AVX2 support (`-mavx2` flag)
- Standard C library

## Running

### Single Input

```bash
make run INPUT_FILE=./data/inputs/graph_V1_D1.bin
```

### All Graphs

```bash
make run_all
```

### Generate Graphs First, Then Run

```bash
make run_fresh
make run_all_fresh
```

### View Help

```bash
make run_help
```

## Results

Output CSV: `./data/results.csv`  
Output binaries: `./data/outputs/`

## Performance Characteristics

### vs. Serial Baseline
- Sparse graphs (density ≤ 0.5): ~2–4× faster due to CSR + SIMD
- Dense graphs (density → 1.0): ~1.5–2× faster (SIMD amortizes less when most edges exist)

### Scaling
- **Single-threaded** — no thread parallelism
- SIMD reduces per-edge compute cost but does not change algorithm iteration count
- Best suited for graphs with 100–10k vertices and moderate density

## Implementation Details

### CSR Format
- `row_ptr[u]` → index into `col_idx[]` and `weights[]` for outgoing edges of u
- `col_idx[i]` → destination vertex v
- `weights[i]` → edge weight w from u to v
- Contiguous arrays enable cache-friendly sequential access and SIMD gather

### SIMD Relaxation Loop
For each source vertex u with outgoing edges:
1. Broadcast `distance[u]` into an AVX2 register (`_mm256_set1_epi32`)
2. For chunks of 8 edges:
   - Load 8 weights, 8 destinations
   - Compute candidate distances: `distance[u] + weight` (vectorized)
   - Compare: `candidate < distance[v]` (vectorized)
   - On any improvement, scalar fallback updates `distance[v]`
3. Handle remaining edges (<8) with scalar loop

## Comparison with Other Versions

| Version | Parallelism | Graph Rep. | Speedup Factor |
|---------|-------------|-----------|----------------|
| Serial baseline | None | Dense (V²) | 1.0× |
| V1_OMP | OpenMP threads | Dense (V²) | 2–6× |
| V2_SIMD | AVX2 SIMD lanes | CSR (sparse) | 2–4× |
| V3_OMP+SIMD | OpenMP + AVX2 | CSR (sparse) | 4–12× |

## Files

- `SIMD_bellmanford.c` — Main SIMD implementation
- `graph_gen.c` — Random graph generator
- `include/graph.h` — CSR graph header
- `include/save_results.h` — Results output header
- `utility/graph.c` — CSR graph utilities
- `utility/save_results.c` — CSV/binary output

## References

1. Gan et al. (2024). "SuperCSR: A Space-Time-Efficient CSR Representation for Large-scale Graph Applications on Supercomputers." ICPP 2024.
2. Dong et al. (2021). "Efficient Stepping Algorithms and Implementations for Parallel Shortest Paths." SPAA 2021, arXiv:2105.06145.

## Notes

- **Architecture-dependent**: Requires AVX2 (Intel Haswell 2013+, AMD Excavator 2015+). Check with `grep avx2 /proc/cpuinfo`.
- **No thread parallelism**: Run this version as a baseline for single-threaded SIMD performance before adding OpenMP (V3_OMP+SIMD).
- **Scalar fallback**: The compare-mask operation skips scalar stores when no improvement is detected, reducing memory traffic on later iterations.

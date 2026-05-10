# V3_OMP+SIMD: Bellman-Ford with OpenMP Frontier Scheduling + AVX2 SIMD

## Overview

This directory prepares the third Bellman-Ford variant for later comparison work. It combines:

1. **OpenMP frontier scheduling** from the V1_OMP version.
2. **CSR graph representation** from the SIMD version.
3. **AVX2 vectorization** inside the relaxation loop.

The goal is to reduce work at both the outer-loop level and the inner-loop level before moving on to benchmarking against the other versions.

## Files

- `bellman_ford_OMP_SIMD.c` - main implementation
- `graph_gen.c` - graph generator
- `include/graph.h` - graph data structures and CSR helpers
- `include/save_results.h` - CSV and output helpers
- `utility/graph.c` - graph utilities
- `utility/save_results.c` - result saving utilities
- `Makefile` - build and run commands

## Build

```bash
make
```

This produces `./build/bellman_ford_OMP_SIMD`.

Requirements:

- GCC with OpenMP support
- AVX2-capable compiler target

## Run

Run a single graph:

```bash
make run THREADS=4
```

Use a specific input file:

```bash
make run INPUT_FILE=./data/inputs/graph_V2_D2.bin THREADS=8
```

Run all generated graphs:

```bash
make run_all THREADS=4
```

Generate graphs first, then run:

```bash
make run_fresh THREADS=4
make run_all_fresh THREADS=4
```

View usage examples:

```bash
make run_help
```

## Output

- CSV results: `./data/results.csv`
- Graph outputs: `./data/outputs/`
- Build artifacts: `./build/`

## Notes

This directory is set up for the V3 implementation only. Comparison targets can be added later once this version is stable and benchmarked against the other variants.
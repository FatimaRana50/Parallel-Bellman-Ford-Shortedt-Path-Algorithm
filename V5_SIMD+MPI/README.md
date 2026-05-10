# V5_SIMD+MPI: Bellman-Ford with MPI + AVX2 SIMD

## Overview

This version combines MPI rank-based work distribution with AVX2 SIMD inside the edge relaxation loop. It uses CSR graph storage, rank-local vertex partitions, and a global update exchange each iteration.

## Build

```bash
make
```

## Generate Graphs

```bash
make generate_graphs
```

This writes the graph inputs into `../parallel_bellman_ford/data/inputs/`.

## Run Locally

```bash
make run NP=4
```

Run every graph:

```bash
make run_all NP=4
```

## Docker Build and Run

Build inside `node0`:

```bash
make docker_build
```

Generate graphs inside Docker:

```bash
make docker_generate_graphs
```

Run one graph through Docker with all MPI ranks from the hostfile:

```bash
make docker_run NP=4 INPUT_FILE=/project/parallel_bellman_ford/data/inputs/graph_V3_D1.bin
```

Run every graph through Docker:

```bash
make docker_run_all NP=4
```

## Output

- Results CSV: `./results.csv`
- Output binaries: `../parallel_bellman_ford/data/outputs/`

## Notes

- `NP` controls the number of MPI ranks.
- The program writes results.csv in the V5 directory.
- The Docker commands assume the container name is `node0` and the hostfile is mounted at `/project/hostfile`.
# V4_MPI: Bellman-Ford with MPI

## Overview

This directory contains the MPI version of Bellman-Ford. Every MPI rank participates in the relaxation loop, but only rank 0 prints the final summary and writes the output files.

The program is meant to be run with `mpirun -np <ranks> ...`.

## Build

```bash
make
```

This builds `./build/v4_mpi` using `mpicc`.

### Docker build

If you are working through the Docker containers, build inside `node0`:

```bash
make docker_build
```

## Run One Graph

```bash
make run NP=4
```

Or run a specific graph:

```bash
make run INPUT_FILE=../parallel_bellman_ford/data/inputs/graph_V3_D1.bin NP=8
```

### Docker run across all ranks

If your MPI containers are named `node0` to `node3` and you want MPI to use them all, run:

```bash
make docker_run INPUT_FILE=/project/parallel_bellman_ford/data/inputs/graph_V3_D1.bin NP=4
```

This uses the hostfile at `/project/hostfile` and launches MPI from `node0`.

To run every graph through Docker:

```bash
make docker_run_all NP=4
```

## Run All Graphs

```bash
make run_all NP=4
```

## Notes

- `NP` controls the number of MPI ranks.
- Rank 0 loads the graph and writes results.
- All ranks participate in the computation; the final output is only centralized on rank 0.
- If a graph has a negative cycle, the verifier is skipped because shortest-path distances are not well-defined.
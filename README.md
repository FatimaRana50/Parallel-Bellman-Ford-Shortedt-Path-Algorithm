# PDC_HYBRID Bellman-Ford Research

This repository collects the Bellman-Ford implementations, benchmark scripts, and analysis artifacts produced during our parallel computing research project. The goal of the study was to measure how far Bellman-Ford can be accelerated by moving from a simple baseline to progressively more advanced combinations of OpenMP, SIMD, and MPI, while keeping the algorithm correct and reproducible across all variants.

The project was not just about adding parallelism. Each version explored a different optimization strategy:

- reducing the amount of work per iteration,
- improving memory layout and cache behavior,
- vectorizing the inner relaxation loop,
- distributing work across threads and processes,
- and comparing the overheads introduced by hybrid execution.

The result is a complete comparison of serial, shared-memory, distributed-memory, and hybrid Bellman-Ford implementations, with the generated CSV files and output folders used for final performance analysis.

## Research Summary

Bellman-Ford is a good candidate for performance experiments because the same relaxation step is repeated many times, which makes the algorithm sensitive to both compute cost and memory traffic. In our study we started with a baseline implementation and then optimized along two axes:

1. Algorithmic work reduction, by avoiding unnecessary edge scans when possible.
2. Hardware utilization, by mapping the work onto OpenMP threads, SIMD lanes, and MPI ranks.

This let us compare not only speedups, but also when each optimization becomes worthwhile. Some approaches helped most on larger graphs, while others introduced enough overhead that they only paid off on certain workloads.

## Main Optimizations We Studied

### V1_OMP: Frontier-Based OpenMP

The first optimization focused on reducing redundant work. Instead of scanning all edges in every iteration, V1 tracks a frontier of active vertices and only relaxes edges that can still improve distances. That change reduces the amount of wasted relaxation work, especially on graphs where many vertices stop changing early.

OpenMP is then used to distribute the remaining relaxation work across threads. This version is our shared-memory baseline for studying thread scaling and scheduling overhead.

### V2_SIMD: CSR + AVX2 Vectorization

The second optimization focuses on memory layout and instruction-level parallelism. The graph is stored in CSR form instead of a dense adjacency matrix, which reduces the amount of data touched during traversal and improves locality for sparse graphs.

On top of CSR, the relaxation loop is vectorized with AVX2 so multiple edge weights can be processed at once. This version is useful for measuring how much performance can be gained from data-level parallelism even without OpenMP or MPI.

### V3_OMP+SIMD: Threading + Vectorization

V3 combines the two ideas above. Frontier-style pruning reduces the work set, OpenMP splits the active work across threads, and AVX2 accelerates the inner relaxation loop. This version shows how shared-memory parallelism and SIMD can complement each other when both the outer loop and inner loop are optimized.

### V4_MPI: Process-Level Distribution

V4 moves the computation to MPI. The work is spread across processes, which makes it possible to study the cost of communication versus the benefit of more parallel workers. This version is especially important for understanding how Bellman-Ford behaves in distributed-memory settings.

### V5_SIMD+MPI: Rank Parallelism with Vectorized Relaxation

V5 combines MPI with AVX2. The motivation here is to keep the distributed-memory decomposition while still improving the cost of each local relaxation step. This version helps isolate whether vectorization still matters once message passing enters the picture.

### V6_MP+MPI: OpenMP + MPI Hybrid

V6 uses both OpenMP and MPI together. Each MPI rank handles a portion of the graph, and OpenMP threads speed up the work inside the rank. This version tests the common hybrid pattern used on multi-node systems with multiple cores per node.

### V7_MP+MPI+SIMD: Full Hybrid

V7 adds SIMD to the OpenMP + MPI combination. It represents the most aggressive optimization in the project and gives the final comparison point for studying whether the added complexity of full hybridization produces a meaningful gain over simpler variants.

## What We Measured

The experiments produce CSV output and per-run result files so that we can compare:

- total execution time,
- Bellman-Ford kernel time,
- the impact of thread count,
- the impact of process count,
- and the tradeoff between locality, parallelism, and communication.

Those outputs feed the analysis scripts in `Analysis/`, including the roofline summaries.

## Project Layout

- `parallel_bellman_ford/` - baseline project and comparison driver for the main experiments.
- `V1_OMP/` - OpenMP-only frontier-based Bellman-Ford.
- `V2_SIMD/` - SIMD-only Bellman-Ford with CSR graph storage.
- `V3_OMP+SIMD/` - OpenMP + SIMD hybrid.
- `V4_MPI/` - MPI-only Bellman-Ford.
- `V5_SIMD+MPI/` - SIMD + MPI hybrid.
- `V6_MP+MPI/` - OpenMP + MPI hybrid.
- `V7_MP+MPI+SIMD/` - OpenMP + MPI + SIMD hybrid.
- `Analysis/` - roofline and performance analysis scripts and generated summaries.
- `All_results/` - collected outputs and final comparison artifacts from the research runs.
- `hostfile` - MPI hostfile used by the Docker-based runs.
- `verify_correctness.py` - correctness checker used to validate outputs.

## Input Graphs

Several targets expect the same six graph files to be present in `parallel_bellman_ford/data/inputs/`:

- `graph_V2_D1.bin`
- `graph_V2_D4.bin`
- `graph_V3_D1.bin`
- `graph_V3_D2.bin`
- `graph_V4_D1.bin`
- `graph_V4_D2.bin`

If that directory does not already contain those files, generate or copy them there once before running the benchmark commands below. The baseline and MPI hybrids read from that shared input directory, so keeping the filenames consistent is important for reproducible runs.

## Baseline Comparison Workflow

The main comparison workflow lives in `parallel_bellman_ford/`.

Run the OpenMP comparison loop across thread counts:

```bash
cd /home/fatima/PDC_HYBRID/parallel_bellman_ford
rm -f data/comparison_results.csv
rm -rf data/comparison_outputs
for t in 1 2 4 8 12; do
  make run_selected_graphs_omp OMP_THREADS=$t
done
```

Run the baseline MPI version in Docker:

```bash
cd /home/fatima/PDC_HYBRID/parallel_bellman_ford
make docker_run_baseline_mpi NP=4
```

Run the baseline hybrid OMP + MPI version in Docker:

```bash
cd /home/fatima/PDC_HYBRID/parallel_bellman_ford
make docker_run_baseline_hybrid NP=4 OMP_THREADS=4
```

Run both baselines back-to-back:

```bash
cd /home/fatima/PDC_HYBRID/parallel_bellman_ford
make docker_run_all_baselines NP=4 OMP_THREADS=4
```

## Version-by-Version Commands

### V1_OMP

Build and run the OpenMP-only version across the chosen thread counts:

```bash
cd /home/fatima/PDC_HYBRID/V1_OMP
rm -f data/results.csv
rm -rf data/outputs
for t in 1 2 4 8 12; do
  make run_selected_graphs THREADS=$t
done
```

### V2_SIMD

Build and run the SIMD-only version:

```bash
cd /home/fatima/PDC_HYBRID/V2_SIMD
rm -f data/results.csv
rm -rf data/outputs
make run_selected_graphs
```

### V3_OMP+SIMD

Build and run the OpenMP + SIMD version across the chosen thread counts:

```bash
cd /home/fatima/PDC_HYBRID/V3_OMP+SIMD
rm -f data/results.csv
rm -rf data/outputs
for t in 1 2 4 8 12; do
  make run_selected_graphs THREADS=$t
done
```

### V4_MPI

Run the MPI-only version:

```bash
cd /home/fatima/PDC_HYBRID/V4_MPI
rm -f results.csv
rm -rf outputs
make run_selected_graphs NP=4
```

### V5_SIMD+MPI

Build the Docker image and run the selected graphs:

```bash
cd /home/fatima/PDC_HYBRID/V5_SIMD+MPI
make docker_build
make docker_run_selected_graphs NP=4
```

### V6_MP+MPI

Build and run the OpenMP + MPI hybrid:

```bash
cd /home/fatima/PDC_HYBRID/V6_MP+MPI
make docker_build
make docker_run_selected_graphs NP=4
```

### V7_MP+MPI+SIMD

Build and run the full hybrid version:

```bash
cd /home/fatima/PDC_HYBRID/V7_MP+MPI+SIMD
make docker_build
make docker_run_selected_graphs NP=4
docker exec node0 cat /project/V7_MP+MPI+SIMD/results.txt | head -10
```

## Reproducibility Notes

- The Docker-based targets assume the container name is `node0` and that the repository is mounted under `/project`.
- MPI targets typically use `NP=4` in the recorded experiments, and the hybrid OpenMP + MPI targets also pass `OMP_THREADS=4` where needed.
- The `data/results.csv`, `results.csv`, and `data/comparison_results.csv` files are the main CSV outputs to inspect after each run.
- Output folders such as `data/outputs/`, `data/comparison_outputs/`, `outputs/`, and the Docker output directories can be removed safely before rerunning experiments.
- If you are reproducing the research from scratch, run the baseline comparisons first, then the V1-V7 variants, and finally the roofline analysis so the summary files reflect the completed benchmark set.

## Analysis

Performance summaries and roofline analysis live under `Analysis/`. The generated per-version summaries and CSV files in that folder are what feed the final research comparison. The analysis stage is where we compare the practical effect of each optimization against the baseline measurements collected from the runs above.

## Execution Order Used In The Research

The typical order for reproducing the full study is:

1. Make sure the six shared input graphs exist in `parallel_bellman_ford/data/inputs/`.
2. Run the baseline comparison workflow in `parallel_bellman_ford/`.
3. Run V1 through V7 using the commands above.
4. Collect the CSV and output artifacts under `All_results/` and `Analysis/`.
5. Use the analysis scripts to generate the final performance summaries and roofline plots.

## Short Project Story

In this research, we used Bellman-Ford as a controlled benchmark to study how different optimization layers interact. The baseline versions show the cost of repeated edge relaxation, while the later versions demonstrate how much of that cost can be hidden or reduced by changing the graph representation, exploiting SIMD instructions, and splitting the work across threads and MPI ranks.

The main takeaway is that there is no single best optimization for every workload. Some versions are strongest when graphs are large enough to amortize parallel overhead, while others are better at reducing the amount of work done in each iteration. The repository therefore serves as both an implementation archive and a reproducibility package for the research results.

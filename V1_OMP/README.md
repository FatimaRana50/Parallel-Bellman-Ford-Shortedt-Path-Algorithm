# V1 — OpenMP Frontier-Based Bellman-Ford

## What This Is
Modified version of the baseline OpenMP Bellman-Ford.
Implements frontier-based scheduling to eliminate redundant edge relaxations.

## Modification Summary
| | Baseline OMP | V1 (This version) |
|---|---|---|
| Per-iteration work | Scans ALL V² edges | Only scans edges from active (frontier) vertices |
| Scheduling | Static chunking of V² array | Dynamic scheduling over active vertex rows |
| Early exit | Exits if zero relaxations | Exits if frontier is empty |

## Paper Justification
Dong, Gu, Sun, Zhang — "Efficient Stepping Algorithms and Implementations
for Parallel Shortest Paths", ACM SPAA 2021, arXiv:2105.06145

## Build
```bash
make
```

## Run
```bash
# First generate a graph using the original repo's graph_gen tool:
./graph_gen <num_vertices> <density> <output_file.bin>

# Then run:
./bf_omp_frontier <graph_file.bin> <results.csv> <output_folder/> <num_threads>
```

## Example
```bash
mkdir -p output
./bf_omp_frontier graph_1000.bin results.csv output/ 4
```

## Expected Output
- `results.csv` — execution time, BF time, thread count, graph info
- `output/v1_omp_frontier_output.txt` — shortest distances from source vertex 0
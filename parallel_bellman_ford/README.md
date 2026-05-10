# Project Overview
This project provides a parallel implementation of the Bellman-Ford algorithm (BF) for finding the shortest paths from a source vertex to all other vertices in a weighted graph. A key advantage of the Bellman-Ford algorithm over alternatives like Dijkstra's is its ability to handle graphs with negative-weighted edges, which is not possible with Dijkstra's implementation.

The project includes three main implementations:


1. Serial Implementation: A single-threaded version that serves as a baseline for performance comparison.


2. Goal-1: OpenMP & MPI: A hybrid approach combining OpenMP for shared-memory parallelism within a single machine and MPI for communication between processes on different machines (distributed memory).

3. Goal-2: OpenMP & CUDA: Another hybrid approach that uses OpenMP to manage the CPU side and offloads computationally intensive tasks to an Nvidia GPU using CUDA.

All implementations are optimized to parallelize the relaxation operations on the graph's edges. The project's performance is evaluated using metrics like overall running time, Bellman-Ford running time, speedup, and efficiency.


## Implementations

### Serial Version

The serial implementation follows a straightforward approach. It initializes distances to all vertices as infinity and the source vertex distance to 0. The core of the algorithm involves iterating n-1 times (where n is the number of vertices) and relaxing each edge. A key optimization is using a single loop to iterate over all edge weights, calculating the from and to vertices on the fly to avoid nested loop overhead. A relax_done variable is also included to allow the algorithm to terminate early if no more relaxations are possible.

### Goal-1: OpenMP & MPI
This hybrid implementation is designed for both shared-memory and distributed-memory systems.

OpenMP (OMP): OpenMP is used to distribute the relaxation operations among threads, which can access shared memory. This approach can be suboptimal for small graphs due to issues like false sharing and the overhead of thread creation and joining, but it performs better on larger graphs.

MPI: MPI is used to distribute graph data across multiple machines or processes. This helps to offload computational work and is particularly useful for very large graphs that might not fit on a single computer. The implementation is designed to minimize communication overhead by only sending necessary updates between nodes. The  FUNNELED multi-threading mode is used, allowing only the master thread of each MPI process to handle communication.

### Goal-2: OpenMP & CUDA
This implementation leverages the parallel processing power of Nvidia GPUs using CUDA. The chosen approach splits the work between the CPU (managed by OpenMP) and the GPU (using CUDA). The GPU handles a portion of the edge relaxation operations in parallel , with the number of weights sent to the device being a configurable parameter via a Command Line Interface (CLI) parameter. The host (CPU) and device (GPU) operations are asynchronous, allowing the CPU to work on its portion of the data while the GPU is busy. After each of the V-1 iterations, the host and device synchronize by exchanging updated distance values.


## Performance 📊
The performance analysis confirms that parallel versions of the algorithm significantly improve execution times on larger graphs.


CUDA is the fastest implementation for graphs from V3 (1,000 vertices) and up. For smaller graphs like V2 (100 vertices), the overhead of data copying to the GPU makes it less efficient than the serial version.

The Goal-1 and Goal-2 hybrid implementations suffered from their distributed memory architecture but were still faster than the serial version. A real-world test on a cluster would likely show better performance for the Goal-1 version.


The only-OMP and only-MPI versions generally perform better on their own than when combined in the Goal-1 implementation.

Performance is sensitive to the graph's size, with a higher number of vertices degrading performance more than an increase in the number of edges.

Different compiler optimization flags also have varied impacts; a higher level of optimization doesn't always lead to the best results. For example, optimization level 1 gave the best results for the Goal-1 version.


## Usage and Reproduction
The project includes a Makefile to simplify compilation and testing. To reproduce all the results from the project report, follow these steps.

* Clean previous runs:

```Bash
make clean
```

* Compile all executables: This will compile the serial, OpenMP, MPI, and CUDA versions with different optimization levels (O0-O3).

```Bash
make all
```

* Generate test graphs: This step creates the graph input files used for the tests.

```Bash
make generate_graphs
```

* Run all tests: This executes each program on the generated graphs and saves the results in the outputs folder.

```Bash
make test
```

* Analyze and plot data: This command requires Python 3 with matplotlib, pandas, and dataframe_image installed.

```Bash
make analyse
```

If you don't have the required Python libraries, you can install them using the provided requirements.txt file:

```Bash
python3 -m pip install -r requirements.txt
```

Individual optimization levels can also be tested with specific commands, such as 

```Bash
make test0 and make test0_cuda.
```
## License
Copyright (C) 2023 Polverino Alessandro 

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.
################################################################################
# QUICK START - OMP VERSION COMPARISON
################################################################################
# Easy targets to compile and run OMP versions for comparison

OMP_OPTIMIZATION_LEVEL ?= 2
OMP_THREADS ?= 4
NP ?= 4
COMPARE_THREADS ?= 1 2 4 8
COMPARISON_RESULTS = ./data/comparison_results.csv
COMPARISON_OUTPUT_DIR = ./data/comparison_outputs
LARGE_COMPARISON_RESULTS = ./data/comparison_results_large.csv
LARGE_COMPARISON_OUTPUT_DIR = ./data/comparison_outputs_large
SELECTED_GRAPH_FILES = graph_V2_D1.bin graph_V2_D4.bin graph_V3_D1.bin graph_V3_D2.bin graph_V4_D1.bin graph_V4_D2.bin

# TEST SIZE OPTIONS: small, medium, large (default: medium)
TEST_SIZE ?= medium
ifeq ($(TEST_SIZE),small)
  GRAPH_FILTER = -name "graph_V1_D1.bin" -o -name "graph_V1_D2.bin" -o -name "graph_V2_D1.bin" -o -name "graph_V2_D2.bin"
  TEST_SIZE_LABEL = SMALL (10, 100 vertices, densities 0.25-0.5)
else ifeq ($(TEST_SIZE),medium)
  GRAPH_FILTER = -name "graph_V1_D*.bin" -o -name "graph_V2_D*.bin"
  TEST_SIZE_LABEL = MEDIUM (10-100 vertices, all densities)
else
  GRAPH_FILTER = -name "graph_*.bin"
  TEST_SIZE_LABEL = LARGE (all graphs: 10-5000 vertices)
endif

# Compile OMP version only (default: -O2 optimization)
compile_omp_only:
	@echo "Compiling OMP-only Bellman-Ford with -O$(OMP_OPTIMIZATION_LEVEL) optimization..."
	mkdir -p ./build
	$(CC) -O$(OMP_OPTIMIZATION_LEVEL) -c -o ./build/graph.o ./src/utility/graph.c
	$(CC) -O$(OMP_OPTIMIZATION_LEVEL) -c -o ./build/save_results.o ./src/utility/save_results.c
	$(CC) -O$(OMP_OPTIMIZATION_LEVEL) -fopenmp -o ./build/only_OMP_bellman-ford_opt$(OMP_OPTIMIZATION_LEVEL) \
		./src/goal-1/only_OMP_bellman-ford.c ./build/graph.o ./build/save_results.o
	@echo "✓ OMP version compiled: ./build/only_OMP_bellman-ford_opt$(OMP_OPTIMIZATION_LEVEL)"

# Initialize comparison environment
init_comparison:
	@mkdir -p $(COMPARISON_OUTPUT_DIR)
	# Ensure the comparison CSV exists and has the correct header. If missing or incorrect, (re)create it.
	@if [ ! -f $(COMPARISON_RESULTS) ] || ! head -n1 $(COMPARISON_RESULTS) | grep -q "^version,optimization_level,program,vertices,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles"; then \
		echo "version,optimization_level,program,vertices,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles" > $(COMPARISON_RESULTS); \
	fi

init_large_comparison:
	@mkdir -p $(LARGE_COMPARISON_OUTPUT_DIR)
	@if [ ! -f $(LARGE_COMPARISON_RESULTS) ] || ! head -n1 $(LARGE_COMPARISON_RESULTS) | grep -q "^version,optimization_level,program,vertices,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles"; then \
		echo "version,optimization_level,program,vertices,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles" > $(LARGE_COMPARISON_RESULTS); \
	fi

# Generate graphs for testing
gen_graphs_omp: compile_omp_only init_comparison
	@echo "Generating test graphs..."
	@mkdir -p ./data/inputs
	@if [ ! -f ./build/graph_generator ]; then \
		$(CC) -O2 -c -o ./build/graph.o ./src/utility/graph.c; \
		$(CC) -O2 -o ./build/graph_generator ./src/graph_gen.c ./build/graph.o; \
	fi
	./build/graph_generator

ensure_graphs_omp:
	@mkdir -p ./data/inputs
	@if ! ls ./data/inputs/graph_*.bin >/dev/null 2>&1; then \
		echo "No input graphs found; generating once..."; \
		$(MAKE) gen_graphs_omp; \
	fi

# Run OMP version from parallel_bellman_ford (Goal 1)
run_omp_parallel: compile_omp_only gen_graphs_omp
	@echo "\n========================================="
	@echo "Running OMP from: parallel_bellman_ford"
	@echo "Optimization Level: $(OMP_OPTIMIZATION_LEVEL)"
	@echo "Threads: $(OMP_THREADS)"
	@echo "Test Size: $(TEST_SIZE_LABEL)"
	@echo "========================================="
	@if [ -d ./data/inputs ] && [ ! -z "$$(find ./data/inputs $(GRAPH_FILTER) 2>/dev/null)" ]; then \
		for input in $$(find ./data/inputs $(GRAPH_FILTER) | sort); do \
			echo "[parallel_bellman_ford] Running on $$input..."; \
			./build/only_OMP_bellman-ford_opt$(OMP_OPTIMIZATION_LEVEL) $$input $(COMPARISON_RESULTS) $(COMPARISON_OUTPUT_DIR) $(OMP_THREADS); \
		done; \
	else \
		echo "ERROR: No graphs matching test size found in ./data/inputs"; \
		exit 1; \
	fi

# Run OMP version from V1_OMP for comparison
run_omp_v1: gen_graphs_omp
	@echo "\n========================================="
	@echo "Running OMP from: V1_OMP"
	@echo "Optimization Level: 2 (V1 default)"
	@echo "Threads: $(OMP_THREADS)"
	@echo "Test Size: $(TEST_SIZE_LABEL)"
	@echo "========================================="
	@cd ../V1_OMP && \
	if [ ! -f ./build/only_OMP_bellman-ford ]; then \
		make all; \
	fi && \
	if [ -d ../parallel_bellman_ford/data/inputs ] && [ ! -z "$$(find ../parallel_bellman_ford/data/inputs $(GRAPH_FILTER) 2>/dev/null)" ]; then \
		for input in $$(find ../parallel_bellman_ford/data/inputs $(GRAPH_FILTER) | sort); do \
			echo "[V1_OMP] Running on $$input..."; \
			./build/only_OMP_bellman-ford $$input ../parallel_bellman_ford/$(COMPARISON_RESULTS) ../parallel_bellman_ford/$(COMPARISON_OUTPUT_DIR) $(OMP_THREADS); \
		done; \
	else \
		echo "ERROR: No graphs matching test size found."; \
		exit 1; \
	fi

# Run only the six selected graphs for the parallel OMP version.
run_selected_graphs_omp: compile_omp_only init_comparison ensure_graphs_omp
	@echo "\n========================================="
	@echo "Running selected graphs from: parallel_bellman_ford"
	@echo "Optimization Level: $(OMP_OPTIMIZATION_LEVEL)"
	@echo "Threads: $(OMP_THREADS)"
	@echo "Graphs: $(SELECTED_GRAPH_FILES)"
	@echo "========================================="
	@for file in $(SELECTED_GRAPH_FILES); do \
		input=./data/inputs/$$file; \
		if [ ! -f "$$input" ]; then \
			echo "ERROR: Missing $$input"; \
			exit 1; \
		fi; \
		echo "[parallel_bellman_ford] Running on $$input..."; \
		./build/only_OMP_bellman-ford_opt$(OMP_OPTIMIZATION_LEVEL) $$input $(COMPARISON_RESULTS) $(COMPARISON_OUTPUT_DIR) $(OMP_THREADS); \
	done

# Fresh comparison for just V1 vs the parallel OpenMP version.
# Clears old comparison results first, then runs every graph for each thread count.
fresh_compare_v1_parallel:
	@echo "Cleaning old comparison results..."
	@rm -f $(COMPARISON_RESULTS)
	@rm -rf $(COMPARISON_OUTPUT_DIR)
	@mkdir -p $(COMPARISON_OUTPUT_DIR)
	@echo "version,optimization_level,program,vertices,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles" > $(COMPARISON_RESULTS)
	@$(MAKE) compile_omp_only gen_graphs_omp TEST_SIZE=large
	@$(MAKE) -C ../V1_OMP all
	@inputs="$$(find ./data/inputs -name "graph_*.bin" | sort)"; \
	if [ -z "$$inputs" ]; then \
		echo "ERROR: No graphs matching test size found in ./data/inputs"; \
		exit 1; \
	fi; \
	for threads in $(COMPARE_THREADS); do \
		echo "\n========================================="; \
		echo "Running parallel_bellman_ford with $$threads threads"; \
		echo "========================================="; \
		for input in $$inputs; do \
			echo "[parallel_bellman_ford] Running on $$input..."; \
			./build/only_OMP_bellman-ford_opt$(OMP_OPTIMIZATION_LEVEL) $$input $(COMPARISON_RESULTS) $(COMPARISON_OUTPUT_DIR) $$threads; \
		done; \
		echo "\n========================================="; \
		echo "Running V1_OMP with $$threads threads"; \
		echo "========================================="; \
		for input in $$inputs; do \
			echo "[V1_OMP] Running on $$input..."; \
			../V1_OMP/build/only_OMP_bellman-ford $$input $(abspath $(COMPARISON_RESULTS)) $(abspath $(COMPARISON_OUTPUT_DIR)) $$threads; \
		done; \
	done

# Run SIMD version from V2_SIMD for comparison
run_simd_v2: gen_graphs_omp
	@echo "\n========================================="
	@echo "Running SIMD from: V2_SIMD"
	@echo "Optimization Level: 2 (with AVX2)"
	@echo "Test Size: $(TEST_SIZE_LABEL)"
	@echo "========================================="
	@cd ../V2_SIMD && \
	if [ ! -f ./build/only_SIMD_bellman-ford ]; then \
		make all; \
	fi && \
	if [ -d ../parallel_bellman_ford/data/inputs ] && [ ! -z "$$(find ../parallel_bellman_ford/data/inputs $(GRAPH_FILTER) 2>/dev/null)" ]; then \
		for input in $$(find ../parallel_bellman_ford/data/inputs $(GRAPH_FILTER) | sort); do \
			echo "[V2_SIMD] Running on $$input..."; \
			./build/only_SIMD_bellman-ford $$input ../parallel_bellman_ford/$(COMPARISON_RESULTS) ../parallel_bellman_ford/$(COMPARISON_OUTPUT_DIR); \
		done; \
	else \
		echo "ERROR: No graphs matching test size found."; \
		exit 1; \
	fi

# Run only the large graphs (V3/V4) for the two OpenMP implementations
run_large_omp_parallel: compile_omp_only init_large_comparison
	@echo "\n========================================="
	@echo "Running LARGE OMP from: parallel_bellman_ford"
	@echo "Optimization Level: $(OMP_OPTIMIZATION_LEVEL)"
	@echo "Threads: $(OMP_THREADS)"
	@echo "Graph Set: V3/V4 only"
	@echo "========================================="
	@if [ -d ./data/inputs ] && [ ! -z "$$(find ./data/inputs -name "graph_V3_*.bin" -o -name "graph_V4_*.bin" 2>/dev/null)" ]; then \
		for input in $$(find ./data/inputs -name "graph_V3_*.bin" -o -name "graph_V4_*.bin" | sort); do \
			echo "[parallel_bellman_ford] Running on $$input..."; \
			./build/only_OMP_bellman-ford_opt$(OMP_OPTIMIZATION_LEVEL) $$input $(LARGE_COMPARISON_RESULTS) $(LARGE_COMPARISON_OUTPUT_DIR) $(OMP_THREADS); \
		done; \
	else \
		echo "ERROR: No V3/V4 graphs found in ./data/inputs"; \
		exit 1; \
	fi

run_large_omp_v1: run_large_omp_parallel
	@echo "\n========================================="
	@echo "Running LARGE OMP from: V1_OMP"
	@echo "Optimization Level: 2 (V1 default)"
	@echo "Threads: $(OMP_THREADS)"
	@echo "Graph Set: V3/V4 only"
	@echo "========================================="
	@cd ../V1_OMP && \
	if [ ! -f ./build/only_OMP_bellman-ford ]; then \
		make all; \
	fi && \
	if [ -d ../parallel_bellman_ford/data/inputs ] && [ ! -z "$$(find ../parallel_bellman_ford/data/inputs -name "graph_V3_*.bin" -o -name "graph_V4_*.bin" 2>/dev/null)" ]; then \
		for input in $$(find ../parallel_bellman_ford/data/inputs -name "graph_V3_*.bin" -o -name "graph_V4_*.bin" | sort); do \
			echo "[V1_OMP] Running on $$input..."; \
			./build/only_OMP_bellman-ford $$input ../parallel_bellman_ford/$(LARGE_COMPARISON_RESULTS) ../parallel_bellman_ford/$(LARGE_COMPARISON_OUTPUT_DIR) $(OMP_THREADS); \
		done; \
	else \
		echo "ERROR: No V3/V4 graphs found."; \
		exit 1; \
	fi

compare_large_v3_v4: run_large_omp_parallel run_large_omp_v1
	@echo "\n========================================="
	@echo "✓ LARGE V3/V4 COMPARISON COMPLETE!"
	@echo "Results saved to: $(LARGE_COMPARISON_RESULTS)"
	@echo "Output files saved to: $(LARGE_COMPARISON_OUTPUT_DIR)"
	@echo "========================================="
	@echo "\nTo analyze results, run:"
	@echo "  cat $(LARGE_COMPARISON_RESULTS)"

################################################################################
# GOAL-1 AND MPI VERSIONS (separate result CSVs)
################################################################################

GOAL1_RESULTS = ./data/goal1_results.csv
GOAL1_OUTPUT_DIR = ./data/goal1_outputs
MPI_RESULTS = ./data/mpi_results.csv
MPI_OUTPUT_DIR = ./data/mpi_outputs

# Initialize goal-1 results
init_goal1:
	@mkdir -p $(GOAL1_OUTPUT_DIR)
	@if [ ! -f $(GOAL1_RESULTS) ] || ! head -n1 $(GOAL1_RESULTS) | grep -q "^threads_num,cuda_threads_num,processes_num"; then \
		echo "threads_num,cuda_threads_num,processes_num,program,vertices_num,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles" > $(GOAL1_RESULTS); \
	fi

# Initialize MPI results
init_mpi:
	@mkdir -p $(MPI_OUTPUT_DIR)
	@if [ ! -f $(MPI_RESULTS) ] || ! head -n1 $(MPI_RESULTS) | grep -q "^threads_num,cuda_threads_num,processes_num"; then \
		echo "threads_num,cuda_threads_num,processes_num,program,vertices_num,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles" > $(MPI_RESULTS); \
	fi

# Compile goal-1 (hybrid OMP+MPI)
compile_goal1: init_goal1
	@echo "Compiling goal_1_bellman-ford (OMP+MPI hybrid)..."
	@mkdir -p ./build
	@if [ ! -f ./build/graph.o ]; then $(CC) -O2 -c -o ./build/graph.o ./src/utility/graph.c; fi
	@if [ ! -f ./build/save_results.o ]; then $(CC) -O2 -c -o ./build/save_results.o ./src/utility/save_results.c; fi
	@mpicc -O2 -fopenmp -o ./build/goal_1_bellman-ford ./src/goal-1/goal_1_bellman-ford.c ./build/graph.o ./build/save_results.o
	@echo "✓ Compiled: ./build/goal_1_bellman-ford"

# Compile MPI-only
compile_mpi: init_mpi
	@echo "Compiling only_MPI_bellman-ford..."
	@mkdir -p ./build
	@if [ ! -f ./build/graph.o ]; then $(CC) -O2 -c -o ./build/graph.o ./src/utility/graph.c; fi
	@if [ ! -f ./build/save_results.o ]; then $(CC) -O2 -c -o ./build/save_results.o ./src/utility/save_results.c; fi
	@mpicc -O2 -o ./build/only_MPI_bellman-ford ./src/goal-1/only_MPI_bellman-ford.c ./build/graph.o ./build/save_results.o
	@echo "✓ Compiled: ./build/only_MPI_bellman-ford"

# Run goal-1 on selected graphs
run_selected_goal1: compile_goal1 ensure_graphs_omp
	@echo "\n========================================="
	@echo "Running goal_1_bellman-ford (OMP+MPI) on selected graphs"
	@echo "Processes (NP): $(NP)"
	@echo "OMP Threads: $(OMP_THREADS)"
	@echo "========================================="
	@for file in $(SELECTED_GRAPH_FILES); do \
		input=./data/inputs/$$file; \
		if [ -f "$$input" ]; then \
			echo "Running on: $$input"; \
			mpirun -np $(NP) --allow-run-as-root --hostfile /project/hostfile --mca btl_tcp_if_include eth0 ./build/goal_1_bellman-ford $$input $(GOAL1_RESULTS) $(GOAL1_OUTPUT_DIR) $(NP) $(OMP_THREADS); \
		fi; \
	done

# Run MPI-only on selected graphs
run_selected_mpi: compile_mpi ensure_graphs_omp
	@echo "\n========================================="
	@echo "Running only_MPI_bellman-ford on selected graphs"
	@echo "Processes (NP): $(NP)"
	@echo "========================================="
	@for file in $(SELECTED_GRAPH_FILES); do \
		input=./data/inputs/$$file; \
		if [ -f "$$input" ]; then \
			echo "Running on: $$input"; \
			mpirun -np $(NP) --allow-run-as-root --hostfile /project/hostfile --mca btl_tcp_if_include eth0 ./build/only_MPI_bellman-ford $$input $(MPI_RESULTS) $(MPI_OUTPUT_DIR) $(NP); \
		fi; \
	done

# Run all three versions on selected graphs (goal1, mpi, omp)
run_all_selected: run_selected_goal1 run_selected_mpi run_selected_graphs_omp
	@echo "\n========================================="
	@echo "✓ ALL THREE VERSIONS COMPLETE!"
	@echo "Goal-1 results: $(GOAL1_RESULTS)"
	@echo "MPI results: $(MPI_RESULTS)"
	@echo "OMP results: $(COMPARISON_RESULTS)"
	@echo "========================================="

# Run ALL THREE versions for side-by-side comparison
compare: run_omp_parallel run_omp_v1 run_simd_v2
	@echo "\n========================================="
	@echo "✓ COMPARISON COMPLETE!"
	@echo "Results saved to: $(COMPARISON_RESULTS)"
	@echo "Output files saved to: $(COMPARISON_OUTPUT_DIR)"
	@echo "========================================="
	@echo "\nTo analyze results, run:"
	@echo "  cat $(COMPARISON_RESULTS)"

# FAST comparison - small graphs only (quick test)
fast_compare:
	@echo "Starting FAST comparison with small graphs...\n"
	# Backup existing comparison CSV (timestamped) so results are preserved
	@if [ -f $(COMPARISON_RESULTS) ]; then \
		cp $(COMPARISON_RESULTS) $(COMPARISON_RESULTS).bak.$$(date +%s); \
		echo "Backed up existing $(COMPARISON_RESULTS) to $(COMPARISON_RESULTS).bak.$$$(date +%s)"; \
	fi
	# Clear only comparison outputs (do not remove whole ./data)
	@mkdir -p $(COMPARISON_OUTPUT_DIR)
	@rm -rf $(COMPARISON_OUTPUT_DIR)/* 2>/dev/null || true
	@$(MAKE) compare TEST_SIZE=small OMP_THREADS=$(OMP_THREADS) OMP_OPTIMIZATION_LEVEL=$(OMP_OPTIMIZATION_LEVEL)

# Quick single run - run one OMP version only
quick_run: compile_omp_only init_comparison
	@echo "Quick run with smallest graph (parallel_bellman_ford only)...\n"
	@mkdir -p ./data/inputs
	@if [ ! -f ./build/graph_generator ]; then \
		$(CC) -O2 -c -o ./build/graph.o ./src/utility/graph.c; \
		$(CC) -O2 -o ./build/graph_generator ./src/graph_gen.c ./build/graph.o; \
	fi
	@./build/graph_generator > /dev/null 2>&1
	@SMALLEST=$$(find ./data/inputs -name "graph_V1_D1.bin" | head -1); \
	if [ -f "$$SMALLEST" ]; then \
		echo "Running on: $$SMALLEST (10 vertices, density 0.25) - FASTEST TEST"; \
		time ./build/only_OMP_bellman-ford_opt$(OMP_OPTIMIZATION_LEVEL) $$SMALLEST $(COMPARISON_RESULTS) $(COMPARISON_OUTPUT_DIR) $(OMP_THREADS); \
	else \
		echo "ERROR: Could not find test graph"; \
		exit 1; \
	fi

# Show usage for comparison targets
help_compare:
	@echo "=== THREE-VERSION COMPARISON TARGETS ==="
	@echo ""
	@echo "VERSIONS COMPARED:"
	@echo "  • parallel_bellman_ford (OpenMP Goal-1)"
	@echo "  • V1_OMP (OpenMP frontier-based)"
	@echo "  • V2_SIMD (Single-threaded AVX2 vectorized)"
	@echo ""
	@echo "MAIN COMMANDS:"
	@echo "  make quick_run                 - Quickest test (1 small graph, parallel_bellman_ford only)"
	@echo "  make fast_compare              - Fast comparison (small graphs, all 3 versions)"
	@echo "  make compare                   - Full comparison (all graphs, all 3 versions)"
	@echo "  make compile_omp_only          - Compile OMP version only"
	@echo ""
	@echo "CLEANING & FRESH RUNS:"
	@echo "  make clean_comparison          - Remove only comparison_results.csv (keeps input graphs)"
	@echo "  make fresh_fast_compare        - Clean CSV + run fast comparison (all 3 versions)"
	@echo "  make fresh_compare             - Clean CSV + run full comparison (all 3 versions)"
	@echo ""
	@echo "TEST SIZE OPTIONS (use with any target):"
	@echo "  TEST_SIZE=small                - 2 graphs: V1 (10v), V2 (100v) @ low/medium density (FASTEST)"
	@echo "  TEST_SIZE=medium               - 8 graphs: V1-V2 @ all densities (DEFAULT)"
	@echo "  TEST_SIZE=large                - 16 graphs: all sizes & densities (SLOWEST)"
	@echo ""
	@echo "TUNING PARAMETERS (OMP versions only):"
	@echo "  OMP_OPTIMIZATION_LEVEL=N       - Compile optimization: 0, 1, 2 (default), 3"
	@echo "  OMP_THREADS=N                  - Thread count (default: 4). V2_SIMD ignores this."
	@echo ""
	@echo "EXAMPLES:"
	@echo "  make quick_run                          # ~10 seconds"
	@echo "  make fast_compare OMP_THREADS=8         # ~1 minute (3 versions, small graphs)"
	@echo "  make fresh_fast_compare                 # Clean CSV, run fast comparison"
	@echo "  make compare TEST_SIZE=small            # ~2 minutes (3 versions)"
	@echo "  make compare TEST_SIZE=medium           # ~15 minutes (3 versions)"
	@echo "  make compare TEST_SIZE=large            # ~60+ minutes (3 versions, all graphs)"
	@echo "  make fresh_compare TEST_SIZE=medium     # Clean and re-run medium test (all 3 versions)"
	@echo "  make fast_compare OMP_OPTIMIZATION_LEVEL=3 OMP_THREADS=12"
	@echo ""
	@echo "NOTE: fast_compare no longer removes all data under ./data."
	@echo "Existing ./data/comparison_results.csv is backed up before a run."
	@echo ""
	@echo "VIEW RESULTS:"
	@echo "  cat ./data/comparison_results.csv       # See all three versions' timing results"
	@echo ""

################################################################################
# ORIGINAL BUILD SYSTEM (Multi-optimization levels & multiple implementations)
################################################################################

resultfile_opt0 = ./data/exec_data_opt0.csv
resultfile_opt1 = ./data/exec_data_opt1.csv
resultfile_opt2 = ./data/exec_data_opt2.csv
resultfile_opt3 = ./data/exec_data_opt3.csv
inputs_dir = ./data/inputs
outputs_dir = ./data/outputs

CC = gcc
mpirunner = mpiexec
mpicompiler = mpicc

all: clean build

build: compile0 compile1 compile2 compile3 

test: test0 test0_cuda test1 test1_cuda test2 test2_cuda test3 test3_cuda

test_only_cuda: test0_cuda test1_cuda test2_cuda test3_cuda

compile0:
	if ! [ -d "./build" ]; then mkdir "./build"; fi

	$(CC) -O0 -c -o ./build/graph.o ./src/utility/graph.c 
	$(CC) -O0 -c -o ./build/save_results.o ./src/utility/save_results.c

	$(CC) -O0 -o ./build/graph_generator ./src/graph_gen.c ./build/graph.o

	$(CC) -O0 -o ./build/bellman-ford-serial_opt0 ./src/serial-version/bellman-ford-serial.c ./build/graph.o ./build/save_results.o 
	$(CC) -O0 -fopenmp -o ./build/only_OMP_bellman-ford_opt0 ./src/goal-1/only_OMP_bellman-ford.c ./build/graph.o ./build/save_results.o 
	$(mpicompiler) -O0 -o ./build/only_MPI_bellman-ford_opt0 ./src/goal-1/only_MPI_bellman-ford.c ./build/graph.o ./build/save_results.o 
	$(mpicompiler) -O0 -fopenmp -o ./build/goal_1_bellman-ford_opt0 ./src/goal-1/goal_1_bellman-ford.c ./build/graph.o ./build/save_results.o 
	nvcc -Xcompiler -O0 -o ./build/only_cuda_bellman-ford_opt0 ./src/goal-2/only_cuda_bellman-ford.cu ./build/graph.o ./build/save_results.o
	nvcc -Xcompiler -fopenmp -O0 -o ./build/goal_2_bellman-ford_opt0 ./src/goal-2/goal_2_bellman-ford.cu ./build/graph.o ./build/save_results.o 
	rm -rf ./build/*.o

compile1:
	if ! [ -d "./build" ]; then mkdir "./build"; fi

	$(CC) -O1 -c -o ./build/graph.o ./src/utility/graph.c 
	$(CC) -O1 -c -o ./build/save_results.o ./src/utility/save_results.c

	$(CC) -O1 -o ./build/graph_generator ./src/graph_gen.c ./build/graph.o

	$(CC) -O1 -o ./build/bellman-ford-serial_opt1 ./src/serial-version/bellman-ford-serial.c ./build/graph.o ./build/save_results.o 
	$(CC) -O1 -fopenmp -o ./build/only_OMP_bellman-ford_opt1 ./src/goal-1/only_OMP_bellman-ford.c ./build/graph.o ./build/save_results.o
	$(mpicompiler) -O1 -o ./build/only_MPI_bellman-ford_opt1 ./src/goal-1/only_MPI_bellman-ford.c ./build/graph.o ./build/save_results.o 
	$(mpicompiler) -O1 -fopenmp -o ./build/goal_1_bellman-ford_opt1 ./src/goal-1/goal_1_bellman-ford.c ./build/graph.o ./build/save_results.o
	nvcc -Xcompiler -O1 -o ./build/only_cuda_bellman-ford_opt1 ./src/goal-2/only_cuda_bellman-ford.cu ./build/graph.o ./build/save_results.o
	nvcc -Xcompiler -fopenmp -O1 -o ./build/goal_2_bellman-ford_opt1 ./src/goal-2/goal_2_bellman-ford.cu ./build/graph.o ./build/save_results.o
	rm -rf ./build/*.o


compile2:
	if ! [ -d "./build" ]; then mkdir "./build"; fi

	$(CC) -O2 -c -o ./build/graph.o ./src/utility/graph.c 
	$(CC) -O2 -c -o ./build/save_results.o ./src/utility/save_results.c

	$(CC) -O2 -o ./build/graph_generator ./src/graph_gen.c ./build/graph.o

	$(CC) -O2 -o ./build/bellman-ford-serial_opt2 ./src/serial-version/bellman-ford-serial.c ./build/graph.o ./build/save_results.o 
	$(CC) -O2 -fopenmp -o ./build/only_OMP_bellman-ford_opt2 ./src/goal-1/only_OMP_bellman-ford.c ./build/graph.o ./build/save_results.o 
	$(mpicompiler) -O2 -o ./build/only_MPI_bellman-ford_opt2 ./src/goal-1/only_MPI_bellman-ford.c ./build/graph.o ./build/save_results.o 
	$(mpicompiler) -O2 -fopenmp -o ./build/goal_1_bellman-ford_opt2 ./src/goal-1/goal_1_bellman-ford.c ./build/graph.o ./build/save_results.o
	nvcc -Xcompiler -O2 -o ./build/only_cuda_bellman-ford_opt2 ./src/goal-2/only_cuda_bellman-ford.cu ./build/graph.o ./build/save_results.o
	nvcc -Xcompiler -fopenmp -O2 -o ./build/goal_2_bellman-ford_opt2 ./src/goal-2/goal_2_bellman-ford.cu ./build/graph.o ./build/save_results.o
	rm -rf ./build/*.o


compile3:
	if ! [ -d "./build" ]; then mkdir "./build"; fi

	$(CC) -O3 -c -o ./build/graph.o ./src/utility/graph.c 
	$(CC) -O3 -c -o ./build/save_results.o ./src/utility/save_results.c

	$(CC) -O3 -o ./build/graph_generator ./src/graph_gen.c ./build/graph.o

	$(CC) -O3 -o ./build/bellman-ford-serial_opt3 ./src/serial-version/bellman-ford-serial.c ./build/graph.o ./build/save_results.o 
	$(CC) -O3 -fopenmp -o ./build/only_OMP_bellman-ford_opt3 ./src/goal-1/only_OMP_bellman-ford.c ./build/graph.o ./build/save_results.o 
	$(mpicompiler) -O3 -Wall -o ./build/only_MPI_bellman-ford_opt3 ./src/goal-1/only_MPI_bellman-ford.c ./build/graph.o ./build/save_results.o 
	$(mpicompiler) -O3 -fopenmp -Wall -o ./build/goal_1_bellman-ford_opt3 ./src/goal-1/goal_1_bellman-ford.c ./build/graph.o ./build/save_results.o 
	nvcc -Xcompiler -O3 -o ./build/only_cuda_bellman-ford_opt3 ./src/goal-2/only_cuda_bellman-ford.cu ./build/graph.o ./build/save_results.o
	nvcc -Xcompiler -fopenmp -O3 -o ./build/goal_2_bellman-ford_opt3 ./src/goal-2/goal_2_bellman-ford.cu ./build/graph.o ./build/save_results.o
	rm -rf ./build/*.o


generate_graphs:
	mkdir -p "./data/inputs"
	./build/graph_generator

test0: generate_graphs
	mkdir -p "./data/outputs"
	if [ ! -f $(resultfile_opt0) ]; then \
		touch $(resultfile_opt0); \
		echo "threads_num,cuda_threads_num,processes_num,program,vertices_num,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles" >> $(resultfile_opt0); \
	fi
	echo "-------------------------------------- \n> INFO: starting tests with optimization level 0"
	for input_filename in	./data/inputs/*.bin; do \
		./build/bellman-ford-serial_opt0 $$input_filename $(resultfile_opt0) $(outputs_dir); \
		echo "> INFO: finished serial execution on $$input_filename"; \
		for num_threads in 1 2 4 8 12; do \
			./build/only_OMP_bellman-ford_opt0 $$input_filename $(resultfile_opt0) $(outputs_dir) $$num_threads; \
			echo "> INFO: finished parallel 'only_OMP_bellman-ford_opt0' execution on $$input_filename with $$num_threads threads"; \
		done; \
		for num_processes in 1 2 4 8; do \
			$(mpirunner) -n $$num_processes ./build/only_MPI_bellman-ford_opt0  $$input_filename $(resultfile_opt0) $(outputs_dir) $$num_threads; \
			echo "> INFO: finished parallel 'only_MPI_bellman-ford_opt0' execution on $$input_filename with $$num_processes processes"; \
		done; \
		for num_threads in 1 2 4 8; do \
			$(mpirunner) -n 2 ./build/goal_1_bellman-ford_opt0 $$input_filename $(resultfile_opt0) $(outputs_dir) $$num_threads; \
			echo "> INFO: finished parallel 'goal_1_bellman-ford_opt0' execution on $$input_filename with 2 processes and $$num_threads threads per process"; \
		done; \
		for num_threads in 1 2 4; do \
			$(mpirunner) -n 4 ./build/goal_1_bellman-ford_opt0 $$input_filename $(resultfile_opt0) $(outputs_dir) $$num_threads; \
			echo "> INFO: finished parallel 'goal_1_bellman-ford_opt0' execution on $$input_filename with 4 processes and $$num_threads threads per process"; \
		done; \
		for num_threads in 1 2; do \
			$(mpirunner) -n 8 ./build/goal_1_bellman-ford_opt0 $$input_filename $(resultfile_opt0) $(outputs_dir) $$num_threads; \
			echo "> INFO: finished parallel 'goal_1_bellman-ford_opt0' execution on $$input_filename with 8 processes and $$num_threads threads per process"; \
		done; \
		echo "> INFO: finished all parallel executions on $$input_filename\n\n"; \
	done;

test0_cuda: generate_graphs
	mkdir -p "./data/outputs"
	if [ ! -f $(resultfile_opt0) ]; then \
		touch $(resultfile_opt0); \
		echo "threads_num,cuda_threads_num,processes_num,program,vertices_num,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles" >> $(resultfile_opt0); \
	fi
	for input_filename in	./data/inputs/*.bin; do \
		./build/only_cuda_bellman-ford_opt0 $$input_filename $(resultfile_opt0) $(outputs_dir); \
		echo "> INFO: finished parallel 'only_cuda_bellman-ford_opt0' execution on $$input_filename"; \
		for num_threads in 1 2 4 8 12; do \
			for perc in 0.25 0.5 0.75; do \
				./build/goal_2_bellman-ford_opt0 $$input_filename $(resultfile_opt0) $(outputs_dir) $$num_threads $$perc; \
				echo "> INFO: finished parallel 'goal_2_bellman-ford_opt0' execution on $$input_filename with $$num_threads threads and $$perc percent on device"; \
			done; \
		done; \
	done;

test1: generate_graphs
	mkdir -p "./data/outputs"
	if [ ! -f $(resultfile_opt1) ]; then \
		touch $(resultfile_opt1); \
		echo "threads_num,cuda_threads_num,processes_num,program,vertices_num,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles" >> $(resultfile_opt1); \
	fi
	echo "-------------------------------------- \n> INFO: starting tests with optimization level 0"
	for input_filename in	./data/inputs/*.bin; do \
		./build/bellman-ford-serial_opt1 $$input_filename $(resultfile_opt1) $(outputs_dir); \
		echo "> INFO: finished serial execution on $$input_filename"; \
		for num_threads in 1 2 4 8 12; do \
			./build/only_OMP_bellman-ford_opt1 $$input_filename $(resultfile_opt1) $(outputs_dir) $$num_threads; \
			echo "> INFO: finished parallel 'only_OMP_bellman-ford_opt1' execution on $$input_filename with $$num_threads threads"; \
		done; \
		for num_processes in 1 2 4 8; do \
			$(mpirunner) -n $$num_processes ./build/only_MPI_bellman-ford_opt1  $$input_filename $(resultfile_opt1) $(outputs_dir) $$num_threads; \
			echo "> INFO: finished parallel 'only_MPI_bellman-ford_opt1' execution on $$input_filename with $$num_processes processes"; \
		done; \
		for num_threads in 1 2 4 8; do \
			$(mpirunner) -n 2 ./build/goal_1_bellman-ford_opt1 $$input_filename $(resultfile_opt1) $(outputs_dir) $$num_threads; \
			echo "> INFO: finished parallel 'goal_1_bellman-ford_opt1' execution on $$input_filename with 2 processes and $$num_threads threads per process"; \
		done; \
		for num_threads in 1 2 4; do \
			$(mpirunner) -n 4 ./build/goal_1_bellman-ford_opt1 $$input_filename $(resultfile_opt1) $(outputs_dir) $$num_threads; \
			echo "> INFO: finished parallel 'goal_1_bellman-ford_opt1' execution on $$input_filename with 4 processes and $$num_threads threads per process"; \
		done; \
		for num_threads in 1 2; do \
			$(mpirunner) -n 8 ./build/goal_1_bellman-ford_opt1 $$input_filename $(resultfile_opt1) $(outputs_dir) $$num_threads; \
			echo "> INFO: finished parallel 'goal_1_bellman-ford_opt1' execution on $$input_filename with 8 processes and $$num_threads threads per process"; \
		done; \
		echo "> INFO: finished all parallel executions on $$input_filename\n\n"; \
	done;

test1_cuda: generate_graphs
	mkdir -p "./data/outputs"
	if [ ! -f $(resultfile_opt1) ]; then \
		touch $(resultfile_opt1); \
		echo "threads_num,cuda_threads_num,processes_num,program,vertices_num,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles" >> $(resultfile_opt1); \
	fi
	for input_filename in	./data/inputs/*.bin; do \
		./build/only_cuda_bellman-ford_opt1 $$input_filename $(resultfile_opt1) $(outputs_dir); \
		echo "> INFO: finished parallel 'only_cuda_bellman-ford_opt1' execution on $$input_filename"; \
		for num_threads in 1 2 4 8 12; do \
		for perc in 0.25 0.5 0.75; do \
				./build/goal_2_bellman-ford_opt1 $$input_filename $(resultfile_opt1) $(outputs_dir) $$num_threads $$perc; \
				echo "> INFO: finished parallel 'goal_2_bellman-ford_opt1' execution on $$input_filename with $$num_threads threads and $$perc percent on device"; \
			done; \
		done; \
	done;

test2: generate_graphs
	mkdir -p "./data/outputs"
	if [ ! -f $(resultfile_opt2) ]; then \
		touch $(resultfile_opt2); \
		echo "threads_num,cuda_threads_num,processes_num,program,vertices_num,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles" >> $(resultfile_opt2); \
	fi
	echo "-------------------------------------- \n> INFO: starting tests with optimization level 0"
	for input_filename in	./data/inputs/*.bin; do \
		./build/bellman-ford-serial_opt2 $$input_filename $(resultfile_opt2) $(outputs_dir); \
		echo "> INFO: finished serial execution on $$input_filename"; \
		for num_threads in 1 2 4 8 12; do \
			./build/only_OMP_bellman-ford_opt2 $$input_filename $(resultfile_opt2) $(outputs_dir) $$num_threads; \
			echo "> INFO: finished parallel 'only_OMP_bellman-ford_opt2' execution on $$input_filename with $$num_threads threads"; \
		done; \
		for num_processes in 1 2 4 8; do \
			$(mpirunner) -n $$num_processes ./build/only_MPI_bellman-ford_opt2  $$input_filename $(resultfile_opt2) $(outputs_dir) $$num_threads; \
			echo "> INFO: finished parallel 'only_MPI_bellman-ford_opt2' execution on $$input_filename with $$num_processes processes"; \
		done; \
		for num_threads in 1 2 4 8; do \
			$(mpirunner) -n 2 ./build/goal_1_bellman-ford_opt2 $$input_filename $(resultfile_opt2) $(outputs_dir) $$num_threads; \
			echo "> INFO: finished parallel 'goal_1_bellman-ford_opt2' execution on $$input_filename with 2 processes and $$num_threads threads per process"; \
		done; \
		for num_threads in 1 2 4; do \
			$(mpirunner) -n 4 ./build/goal_1_bellman-ford_opt2 $$input_filename $(resultfile_opt2) $(outputs_dir) $$num_threads; \
			echo "> INFO: finished parallel 'goal_1_bellman-ford_opt2' execution on $$input_filename with 4 processes and $$num_threads threads per process"; \
		done; \
		for num_threads in 1 2; do \
			$(mpirunner) -n 8 ./build/goal_1_bellman-ford_opt2 $$input_filename $(resultfile_opt2) $(outputs_dir) $$num_threads; \
			echo "> INFO: finished parallel 'goal_1_bellman-ford_opt2' execution on $$input_filename with 8 processes and $$num_threads threads per process"; \
		done; \
		echo "> INFO: finished all parallel executions on $$input_filename\n\n"; \
	done;

test2_cuda: generate_graphs
	mkdir -p "./data/outputs"
	if [ ! -f $(resultfile_opt2) ]; then \
		touch $(resultfile_opt2); \
		echo "threads_num,cuda_threads_num,processes_num,program,vertices_num,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles" >> $(resultfile_opt2); \
	fi
	for input_filename in	./data/inputs/*.bin; do \
		./build/only_cuda_bellman-ford_opt2 $$input_filename $(resultfile_opt2) $(outputs_dir); \
		echo "> INFO: finished parallel 'only_cuda_bellman-ford_opt2' execution on $$input_filename"; \
		for num_threads in 1 2 4 8 12; do \
		for perc in 0.25 0.5 0.75; do \
				./build/goal_2_bellman-ford_opt2 $$input_filename $(resultfile_opt2) $(outputs_dir) $$num_threads $$perc; \
				echo "> INFO: finished parallel 'goal_2_bellman-ford_opt2' execution on $$input_filename with $$num_threads threads and $$perc percent on device"; \
			done; \
		done; \
	done;

test3: generate_graphs
	mkdir -p "./data/outputs"
	if [ ! -f $(resultfile_opt3) ]; then \
		touch $(resultfile_opt3); \
		echo "threads_num,cuda_threads_num,processes_num,program,vertices_num,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles" >> $(resultfile_opt3); \
	fi
	echo "-------------------------------------- \n> INFO: starting tests with optimization level 0"
	for input_filename in	./data/inputs/*.bin; do \
		./build/bellman-ford-serial_opt3 $$input_filename $(resultfile_opt3) $(outputs_dir); \
		echo "> INFO: finished serial execution on $$input_filename"; \
		for num_threads in 1 2 4 8 12; do \
			./build/only_OMP_bellman-ford_opt3 $$input_filename $(resultfile_opt3) $(outputs_dir) $$num_threads; \
			echo "> INFO: finished parallel 'only_OMP_bellman-ford_opt3' execution on $$input_filename with $$num_threads threads"; \
		done; \
		for num_processes in 1 2 4 8; do \
			$(mpirunner) -n $$num_processes ./build/only_MPI_bellman-ford_opt3  $$input_filename $(resultfile_opt3) $(outputs_dir) $$num_threads; \
			echo "> INFO: finished parallel 'only_MPI_bellman-ford_opt3' execution on $$input_filename with $$num_processes processes"; \
		done; \
		for num_threads in 1 2 4 8; do \
			$(mpirunner) -n 2 ./build/goal_1_bellman-ford_opt3 $$input_filename $(resultfile_opt3) $(outputs_dir) $$num_threads; \
			echo "> INFO: finished parallel 'goal_1_bellman-ford_opt3' execution on $$input_filename with 2 processes and $$num_threads threads per process"; \
		done; \
		for num_threads in 1 2 4; do \
			$(mpirunner) -n 4 ./build/goal_1_bellman-ford_opt3 $$input_filename $(resultfile_opt3) $(outputs_dir) $$num_threads; \
			echo "> INFO: finished parallel 'goal_1_bellman-ford_opt3' execution on $$input_filename with 4 processes and $$num_threads threads per process"; \
		done; \
		for num_threads in 1 2; do \
			$(mpirunner) -n 8 ./build/goal_1_bellman-ford_opt3 $$input_filename $(resultfile_opt3) $(outputs_dir) $$num_threads; \
			echo "> INFO: finished parallel 'goal_1_bellman-ford_opt3' execution on $$input_filename with 8 processes and $$num_threads threads per process"; \
		done; \
		echo "> INFO: finished all parallel executions on $$input_filename\n\n"; \
	done;

test3_cuda: generate_graphs
	mkdir -p "./data/outputs"
	if [ ! -f $(resultfile_opt3) ]; then \
		touch $(resultfile_opt3); \
		echo "threads_num,cuda_threads_num,processes_num,program,vertices_num,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles" >> $(resultfile_opt3); \
	fi
	for input_filename in	./data/inputs/*.bin; do \
		./build/only_cuda_bellman-ford_opt3 $$input_filename $(resultfile_opt3) $(outputs_dir); \
		echo "> INFO: finished parallel 'only_cuda_bellman-ford_opt3' execution on $$input_filename"; \
		for num_threads in 1 2 4 8 12; do \
		for perc in 0.25 0.5 0.75; do \
				./build/goal_2_bellman-ford_opt3 $$input_filename $(resultfile_opt3) $(outputs_dir) $$num_threads $$perc; \
				echo "> INFO: finished parallel 'goal_2_bellman-ford_opt3' execution on $$input_filename with $$num_threads threads and $$perc percent on device"; \
			done; \
		done; \
	done;

analyse:
	mkdir -p "./data/analytics"
	mkdir -p "./data/plots"
	mkdir -p "./data/tables"
	python ./src/analysis/main.py

################################################################################
# CLEANUP TARGETS
################################################################################

clean: clean_build clean_output
	@echo "✓ Cleanup complete"

clean_build:
	rm -rf ./build/*

clean_output:
	rm -rf ./data/*

# Clean only comparison results (keeps input graphs)
clean_comparison:
	@rm -f $(COMPARISON_RESULTS) $(COMPARISON_RESULTS).bak.*
	@rm -rf $(COMPARISON_OUTPUT_DIR)/*
	@echo "✓ Comparison results cleaned"

# Fresh comparison: clean CSV and re-run
fresh_compare: clean_comparison
	@$(MAKE) compare TEST_SIZE=$(TEST_SIZE) OMP_THREADS=$(OMP_THREADS) OMP_OPTIMIZATION_LEVEL=$(OMP_OPTIMIZATION_LEVEL)

# Fresh fast comparison
fresh_fast_compare: clean_comparison
	@$(MAKE) fast_compare OMP_THREADS=$(OMP_THREADS) OMP_OPTIMIZATION_LEVEL=$(OMP_OPTIMIZATION_LEVEL)

################################################################################
# HELP & PHONY TARGETS
################################################################################

help: help_compare
	@echo ""
	@echo "=== FULL BUILD SYSTEM (All Implementations & Optimizations) ==="
	@echo ""
	@echo "make all                       - Full build (compile all optimizations)"
	@echo "make build                     - Compile all (opt0, opt1, opt2, opt3)"
	@echo "make test                      - Run all tests"
	@echo "make generate_graphs           - Generate test input graphs"
	@echo "make analyse                   - Run analysis on results"
	@echo "make clean                     - Clean build and output"
	@echo ""

.PHONY: all build test test_only_cuda compile0 compile1 compile2 compile3 \
	generate_graphs test0 test0_cuda test1 test1_cuda test2 test2_cuda \
	test3 test3_cuda analyse clean clean_build clean_output clean_comparison \
	compile_omp_only init_comparison gen_graphs_omp \
	run_omp_parallel run_omp_v1 compare fast_compare fresh_compare fresh_fast_compare \
	quick_run help help_compare
################################################################################
# DOCKER TARGETS — run baseline MPI versions inside Docker containers
# Usage:
#   make docker_run_baseline_mpi     NP=4
#   make docker_run_baseline_hybrid  NP=4 OMP_THREADS=4
#   make docker_run_all_baselines    NP=4 OMP_THREADS=4
################################################################################

DOCKER_CONTAINER ?= node0
DOCKER_HOSTFILE  ?= /project/hostfile
DOCKER_BF_DIR    ?= /project/parallel_bellman_ford
DOCKER_GRAPHS    = graph_V2_D1.bin graph_V2_D4.bin graph_V3_D1.bin \
                   graph_V3_D2.bin graph_V4_D1.bin graph_V4_D2.bin
BASELINE_MPI_CSV = $(DOCKER_BF_DIR)/data/baseline_mpi_results.csv
BASELINE_MPI_OUT = $(DOCKER_BF_DIR)/data/baseline_mpi_outputs
BASELINE_HYB_CSV = $(DOCKER_BF_DIR)/data/baseline_goal1_results.csv
BASELINE_HYB_OUT = $(DOCKER_BF_DIR)/data/baseline_goal1_outputs
BASELINE_MPI_BIN = $(DOCKER_BF_DIR)/build/only_MPI_bellman-ford
BASELINE_HYB_BIN = $(DOCKER_BF_DIR)/build/goal_1_bellman-ford
MPI_RUN          = mpirun -np $(NP) --allow-run-as-root \
                       --hostfile $(DOCKER_HOSTFILE) \
                       --mca btl_tcp_if_include eth0

docker_init_baseline:
	docker exec $(DOCKER_CONTAINER) bash -c "\
		mkdir -p $(BASELINE_MPI_OUT) $(BASELINE_HYB_OUT) && \
		if [ ! -f $(BASELINE_MPI_CSV) ]; then \
			echo 'threads_num,cuda_threads_num,processes_num,program,vertices_num,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles' \
				> $(BASELINE_MPI_CSV); \
		fi && \
		if [ ! -f $(BASELINE_HYB_CSV) ]; then \
			echo 'threads_num,cuda_threads_num,processes_num,program,vertices_num,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles' \
				> $(BASELINE_HYB_CSV); \
		fi"

docker_run_baseline_mpi: docker_init_baseline
	@echo "=== Running baseline only_MPI_bellman-ford | NP=$(NP) ==="
	docker exec $(DOCKER_CONTAINER) bash -c "\
		for input in $(addprefix $(DOCKER_BF_DIR)/data/inputs/,$(DOCKER_GRAPHS)); do \
			echo Running \$$input; \
			$(MPI_RUN) $(BASELINE_MPI_BIN) \$$input $(BASELINE_MPI_CSV) $(BASELINE_MPI_OUT); \
		done"
	@echo "=== Done. Results: ~/PDC_HYBRID/parallel_bellman_ford/data/baseline_mpi_results.csv ==="

docker_run_baseline_hybrid: docker_init_baseline
	@echo "=== Running baseline goal_1_bellman-ford (OMP+MPI) | NP=$(NP) OMP_THREADS=$(OMP_THREADS) ==="
	docker exec $(DOCKER_CONTAINER) bash -c "\
		for input in $(addprefix $(DOCKER_BF_DIR)/data/inputs/,$(DOCKER_GRAPHS)); do \
			echo Running \$$input; \
			$(MPI_RUN) $(BASELINE_HYB_BIN) \$$input $(BASELINE_HYB_CSV) $(BASELINE_HYB_OUT) $(OMP_THREADS); \
		done"
	@echo "=== Done. Results: ~/PDC_HYBRID/parallel_bellman_ford/data/baseline_goal1_results.csv ==="

docker_run_all_baselines: docker_run_baseline_mpi docker_run_baseline_hybrid
	@echo ""
	@echo "======================================="
	@echo "All baseline MPI versions complete!"
	@echo "Results saved to:"
	@echo "  ~/PDC_HYBRID/parallel_bellman_ford/data/baseline_results.csv"
	@echo "View with:"
	@echo "  column -t -s',' ~/PDC_HYBRID/parallel_bellman_ford/data/baseline_results.csv"
	@echo "======================================="

.PHONY: docker_init_baseline docker_run_baseline_mpi docker_run_baseline_hybrid docker_run_all_baselines
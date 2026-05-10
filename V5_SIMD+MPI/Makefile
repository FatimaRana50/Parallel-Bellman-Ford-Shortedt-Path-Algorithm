CC = mpicc
CFLAGS = -O2 -mavx2
INCLUDES = -I./include
LDFLAGS =

DOCKER_CONTAINER ?= node0
DOCKER_WORKDIR ?= /project/V5_SIMD+MPI
DOCKER_HOSTFILE ?= /project/hostfile
DOCKER_INPUT_DIR ?= /project/parallel_bellman_ford/data/inputs
DOCKER_RESULTS_FILE ?= /project/V5_SIMD+MPI/results.csv
DOCKER_OUTPUT_DIR ?= /project/V5_SIMD+MPI/outputs

BUILD_DIR = ./build
SRC_DIR = .
UTIL_DIR = ./utility

TARGET = $(BUILD_DIR)/v5_mpi_simd
GRAPH_GEN = ./graph_gen

INPUT_FILE ?= ../parallel_bellman_ford/data/inputs/graph_V1_D1.bin
RESULTS_FILE ?= ./results.csv
OUTPUT_DIR ?= ./outputs
NP ?= 4

SELECTED_GRAPH_FILES = graph_V2_D1.bin graph_V2_D4.bin graph_V3_D1.bin graph_V3_D2.bin graph_V4_D1.bin graph_V4_D2.bin

SRCS = bellman_ford_MPI_SIMD.c $(UTIL_DIR)/graph.c $(UTIL_DIR)/save_results.c
OBJS = $(BUILD_DIR)/bellman_ford_MPI_SIMD.o $(BUILD_DIR)/graph.o $(BUILD_DIR)/save_results.o

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(GRAPH_GEN): graph_gen.c $(UTIL_DIR)/graph.c | $(BUILD_DIR)
	$(CC) -O2 $(INCLUDES) -o $(GRAPH_GEN) graph_gen.c $(UTIL_DIR)/graph.c

generate_graphs: $(GRAPH_GEN)
	mkdir -p ../parallel_bellman_ford/data/inputs
	cd ../parallel_bellman_ford && ../V5_SIMD+MPI/$(GRAPH_GEN)

init_run:
	mkdir -p $(OUTPUT_DIR)
	if [ ! -f $(RESULTS_FILE) ]; then \
		echo "threads_num,cuda_threads_num,processes_num,program,vertices_num,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles" > $(RESULTS_FILE); \
	fi

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(UTIL_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

run: $(TARGET) init_run
	@if [ ! -f "$(INPUT_FILE)" ]; then \
		echo "Input file not found: $(INPUT_FILE)"; \
		exit 1; \
	fi
	mkdir -p $(OUTPUT_DIR)
	mpirun -np $(NP) --allow-run-as-root $(TARGET) $(INPUT_FILE) $(RESULTS_FILE) $(OUTPUT_DIR)

run_all: $(TARGET) init_run
	@if ! ls ../parallel_bellman_ford/data/inputs/*.bin >/dev/null 2>&1; then \
		echo "No input graphs found in ../parallel_bellman_ford/data/inputs"; \
		exit 1; \
	fi
	mkdir -p $(OUTPUT_DIR)
	for input in ../parallel_bellman_ford/data/inputs/*.bin; do \
		echo "Running $$input"; \
		mpirun -np $(NP) --allow-run-as-root $(TARGET) $$input $(RESULTS_FILE) $(OUTPUT_DIR); \
	done

run_selected_graphs: $(TARGET) init_run
	@if ! ls ../parallel_bellman_ford/data/inputs/*.bin >/dev/null 2>&1; then \
		echo "No input graphs found in ../parallel_bellman_ford/data/inputs"; \
		exit 1; \
	fi
	mkdir -p $(OUTPUT_DIR)
	for file in $(SELECTED_GRAPH_FILES); do \
		input=../parallel_bellman_ford/data/inputs/$$file; \
		if [ ! -f "$$input" ]; then \
			echo "Input file not found: $$input"; \
			exit 1; \
		fi; \
		echo "Running $$input"; \
		mpirun -np $(NP) --allow-run-as-root --hostfile /project/hostfile --mca btl_tcp_if_include eth0 $(TARGET) $$input $(RESULTS_FILE) $(OUTPUT_DIR); \
	done

docker_build:
	docker exec $(DOCKER_CONTAINER) bash -c "cd /project/V5_SIMD+MPI && mpicc -O2 -mavx2 -o v5_mpi_simd bellman_ford_MPI_SIMD.c utility/graph.c utility/save_results.c -I./include"

docker_run_selected_graphs:
	docker exec $(DOCKER_CONTAINER) bash -c "mkdir -p $(DOCKER_OUTPUT_DIR) && \
		if [ ! -f $(DOCKER_RESULTS_FILE) ]; then \
			echo 'threads_num,cuda_threads_num,processes_num,program,vertices_num,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles' > $(DOCKER_RESULTS_FILE); \
		fi && \
		cd $(DOCKER_WORKDIR) && \
		for input in $(DOCKER_INPUT_DIR)/graph_V2_D1.bin $(DOCKER_INPUT_DIR)/graph_V2_D4.bin $(DOCKER_INPUT_DIR)/graph_V3_D1.bin $(DOCKER_INPUT_DIR)/graph_V3_D2.bin $(DOCKER_INPUT_DIR)/graph_V4_D1.bin $(DOCKER_INPUT_DIR)/graph_V4_D2.bin; do \
			echo Running \$$input; \
			mpirun -np $(NP) --allow-run-as-root --hostfile $(DOCKER_HOSTFILE) --mca btl_tcp_if_include eth0 ./v5_mpi_simd \$$input $(DOCKER_RESULTS_FILE) $(DOCKER_OUTPUT_DIR); \
		done"

clean:
	 rm -rf $(BUILD_DIR) $(TARGET) $(OUTPUT_DIR) $(RESULTS_FILE)

.PHONY: all clean generate_graphs run run_all run_selected_graphs docker_build docker_run_selected_graphs

CC = mpicc
CFLAGS = -O2
INCLUDES = -I./include
LDFLAGS =

DOCKER_CONTAINER ?= node0
DOCKER_WORKDIR ?= /project/V4_MPI
DOCKER_HOSTFILE ?= /project/hostfile
DOCKER_INPUT_DIR ?= /project/parallel_bellman_ford/data/inputs
DOCKER_RESULTS_FILE ?= /project/V4_MPI/results.csv
DOCKER_OUTPUT_DIR ?= /project/V4_MPI/outputs

BUILD_DIR = ./build
SRC_DIR = .
UTIL_DIR = ./utility

TARGET = $(BUILD_DIR)/v4_mpi
GRAPH_GEN = $(BUILD_DIR)/graph_gen

INPUT_FILE ?= ../parallel_bellman_ford/data/inputs/graph_V1_D1.bin
RESULTS_FILE ?= ./results.csv
OUTPUT_DIR ?= ./outputs
NP ?= 4
SELECTED_GRAPH_FILES = graph_V2_D1.bin graph_V2_D4.bin graph_V3_D1.bin graph_V3_D2.bin graph_V4_D1.bin graph_V4_D2.bin

SRCS = $(SRC_DIR)/bellman_ford_onlu_mpi.c $(UTIL_DIR)/graph.c $(UTIL_DIR)/save_results.c
OBJS = $(BUILD_DIR)/bellman_ford_onlu_mpi.o $(BUILD_DIR)/graph.o $(BUILD_DIR)/save_results.o

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(GRAPH_GEN): graph_gen.c $(UTIL_DIR)/graph.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(GRAPH_GEN) graph_gen.c $(UTIL_DIR)/graph.c

generate_graphs: $(GRAPH_GEN)
	mkdir -p ../parallel_bellman_ford/data/inputs
	cd ../parallel_bellman_ford && ./V4_MPI/$(GRAPH_GEN)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(UTIL_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

run: $(TARGET)
	@if [ ! -f "$(INPUT_FILE)" ]; then \
		echo "Input file not found: $(INPUT_FILE)"; \
		exit 1; \
	fi
	mkdir -p $(OUTPUT_DIR)
	mpirun -np $(NP) --allow-run-as-root $(TARGET) $(INPUT_FILE) $(RESULTS_FILE) $(OUTPUT_DIR)

run_selected_graphs: $(TARGET)
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
		mpirun -np $(NP) --allow-run-as-root --hostfile /project/hostfile $(TARGET) $$input $(RESULTS_FILE) $(OUTPUT_DIR); \
	done

docker_build:
	docker exec $(DOCKER_CONTAINER) bash -c "cd /project/V4_MPI && mpicc -O2 -o v4_mpi bellman_ford_onlu_mpi.c utility/graph.c utility/save_results.c -I./include"

docker_run:
	docker exec $(DOCKER_CONTAINER) bash -c "mkdir -p $(DOCKER_OUTPUT_DIR) && \
		if [ ! -f /project/V4_MPI/results.csv ]; then \
			echo 'threads_num,cuda_threads_num,processes_num,program,vertices_num,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles' > /project/V4_MPI/results.csv; \
		fi && \
		cd /project/V4_MPI && \
		mpirun -np $(NP) --allow-run-as-root --hostfile /project/hostfile ./v4_mpi /project/parallel_bellman_ford/data/inputs/graph_V3_D1.bin /project/V4_MPI/results.csv $(DOCKER_OUTPUT_DIR)"

docker_run_selected_graphs:
	docker exec $(DOCKER_CONTAINER) bash -c "mkdir -p $(DOCKER_OUTPUT_DIR) && \
		if [ ! -f /project/V4_MPI/results.csv ]; then \
			echo 'threads_num,cuda_threads_num,processes_num,program,vertices_num,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles' > /project/V4_MPI/results.csv; \
		fi && \
		cd /project/V4_MPI && \
		for input in /project/parallel_bellman_ford/data/inputs/graph_V2_D1.bin /project/parallel_bellman_ford/data/inputs/graph_V2_D4.bin /project/parallel_bellman_ford/data/inputs/graph_V3_D1.bin /project/parallel_bellman_ford/data/inputs/graph_V3_D2.bin /project/parallel_bellman_ford/data/inputs/graph_V4_D1.bin /project/parallel_bellman_ford/data/inputs/graph_V4_D2.bin; do \
			echo Running \$$input; \
			mpirun -np $(NP) --allow-run-as-root --hostfile /project/hostfile ./v4_mpi \$$input /project/V4_MPI/results.csv $(DOCKER_OUTPUT_DIR); \
		done"

docker_run_all:
	docker exec $(DOCKER_CONTAINER) bash -c "mkdir -p $(DOCKER_OUTPUT_DIR) && \
		if [ ! -f /project/V4_MPI/results.csv ]; then \
			echo 'threads_num,cuda_threads_num,processes_num,program,vertices_num,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles' > /project/V4_MPI/results.csv; \
		fi && \
		cd /project/V4_MPI && \
		for input in /project/parallel_bellman_ford/data/inputs/*.bin; do \
			echo Running \$$input; \
			mpirun -np $(NP) --allow-run-as-root --hostfile /project/hostfile ./v4_mpi \$$input /project/V4_MPI/results.csv $(DOCKER_OUTPUT_DIR); \
		done"

run_all: $(TARGET)
	@if ! ls ../parallel_bellman_ford/data/inputs/*.bin >/dev/null 2>&1; then \
		echo "No input graphs found in ../parallel_bellman_ford/data/inputs"; \
		exit 1; \
	fi
	mkdir -p $(OUTPUT_DIR)
	for input in ../parallel_bellman_ford/data/inputs/*.bin; do \
		echo "Running $$input"; \
		mpirun -np $(NP) --allow-run-as-root $(TARGET) $$input $(RESULTS_FILE) $(OUTPUT_DIR); \
	done

run_help:
	@echo "Examples:"
	@echo "  make"
	@echo "  make run NP=4"
	@echo "  make run INPUT_FILE=../parallel_bellman_ford/data/inputs/graph_V3_D1.bin NP=8"
	@echo "  make run_all NP=4"

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all run run_all run_selected_graphs run_help clean generate_graphs docker_build docker_run docker_run_all docker_run_selected_graphs
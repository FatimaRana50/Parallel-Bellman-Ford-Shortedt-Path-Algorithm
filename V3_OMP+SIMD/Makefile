CC = gcc
CFLAGS = -O2 -mavx2 -fopenmp
INCLUDES = -I./include
LDFLAGS = -fopenmp

BUILD_DIR = ./build
SRC_DIR = .
UTIL_DIR = ./utility

SRCS = $(SRC_DIR)/bellman_ford_OMP_SIMD.c $(UTIL_DIR)/graph.c $(UTIL_DIR)/save_results.c
OBJS = $(BUILD_DIR)/bellman_ford_OMP_SIMD.o $(BUILD_DIR)/graph.o $(BUILD_DIR)/save_results.o

TARGET = $(BUILD_DIR)/bellman_ford_OMP_SIMD
GRAPH_GEN = ./graph_gen

INPUT_FILE ?= ./data/inputs/graph_V1_D1.bin
RESULTS_FILE ?= ./data/results.csv
OUTPUT_DIR ?= ./data/outputs
THREADS ?= 4
SELECTED_GRAPH_FILES = graph_V2_D1.bin graph_V2_D4.bin graph_V3_D1.bin graph_V3_D2.bin graph_V4_D1.bin graph_V4_D2.bin

all: $(TARGET)

$(GRAPH_GEN): graph_gen.c $(UTIL_DIR)/graph.c
	$(CC) -O2 $(INCLUDES) -o $(GRAPH_GEN) graph_gen.c $(UTIL_DIR)/graph.c

generate_graphs: $(GRAPH_GEN)
	mkdir -p ./data/inputs
	$(GRAPH_GEN)

init_run:
	mkdir -p $(OUTPUT_DIR)
	if [ ! -f $(RESULTS_FILE) ]; then \
		echo "threads_num,cuda_threads_num,processes_num,program,vertices_num,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles" > $(RESULTS_FILE); \
	fi

run: $(TARGET) init_run
	@if [ ! -f "$(INPUT_FILE)" ]; then \
		echo "Input file not found: $(INPUT_FILE)"; \
		echo "Run 'make generate_graphs' once, or set INPUT_FILE=..."; \
		exit 1; \
	fi
	$(TARGET) $(INPUT_FILE) $(RESULTS_FILE) $(OUTPUT_DIR) $(THREADS)

run_all: $(TARGET) init_run
	@if ! ls ./data/inputs/*.bin >/dev/null 2>&1; then \
		echo "No input graphs found in ./data/inputs"; \
		echo "Run 'make generate_graphs' first."; \
		exit 1; \
	fi
	for input in ./data/inputs/*.bin; do \
		$(TARGET) $$input $(RESULTS_FILE) $(OUTPUT_DIR) $(THREADS); \
	done

run_selected_graphs: $(TARGET) init_run
	@for file in $(SELECTED_GRAPH_FILES); do \
		input=./data/inputs/$$file; \
		if [ ! -f "$$input" ]; then \
			echo "ERROR: Missing $$input"; \
			exit 1; \
		fi; \
		echo "Running on $$input"; \
		$(TARGET) $$input $(RESULTS_FILE) $(OUTPUT_DIR) $(THREADS); \
	done

run_fresh: generate_graphs run

run_all_fresh: generate_graphs run_all

run_help:
	@echo "Examples:"
	@echo "  make run"
	@echo "  make run THREADS=8"
	@echo "  make run INPUT_FILE=./data/inputs/graph_V2_D2.bin THREADS=12"
	@echo "  make run_all THREADS=4"
	@echo "  make run_fresh THREADS=4"
	@echo "  make run_all_fresh THREADS=4"
	@echo "  make generate_graphs"

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(UTIL_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean generate_graphs init_run run run_all run_fresh run_all_fresh run_help
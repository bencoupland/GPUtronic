# GPUTronic — Makefile (Gold 1.0.0)
# Target: NVIDIA Blackwell RTX 5080 (sm_120), CUDA 12.8

NVCC = /usr/local/cuda-12.8/bin/nvcc
ARCH = -arch=sm_120
OPTIMIZATION = -O3
COMMON_FLAGS = $(ARCH) $(OPTIMIZATION) -Iinclude -Xcompiler "-Wall -Wextra" -lineinfo --use_fast_math
LIBS = -lm -lpthread
BUILD_DIR = build

.PHONY: all gold lib demo watch check dyno step run transfer transfer-check episode episode-check episode-proof show clean help

all: gold lib demo

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)
	@mkdir -p results

# ---- Gold reference binary (self-contained CLI) ----
gold: $(BUILD_DIR)/gputronic_gold

$(BUILD_DIR)/gputronic_gold: src/gputronic_gold.cu include/gputronic.h | $(BUILD_DIR)
	@echo "[GPUTronic] Building Gold 1.0.0..."
	$(NVCC) $(COMMON_FLAGS) -o $@ src/gputronic_gold.cu $(LIBS)
	@echo "[GPUTronic] Built: $@"

# ---- Static library (same translation unit exports the C API) ----
lib: $(BUILD_DIR)/libgputronic_gold.a

$(BUILD_DIR)/libgputronic_gold.a: src/gputronic_gold.cu include/gputronic.h | $(BUILD_DIR)
	@echo "[GPUTronic] Building libgputronic_gold.a..."
	$(NVCC) $(COMMON_FLAGS) -c -o $(BUILD_DIR)/gputronic_gold.o src/gputronic_gold.cu
	# Strip main for the library object: rebuild with -DGPUTRONIC_NO_MAIN
	$(NVCC) $(COMMON_FLAGS) -DGPUTRONIC_NO_MAIN -c -o $(BUILD_DIR)/gputronic_gold_lib.o src/gputronic_gold.cu
	ar rcs $@ $(BUILD_DIR)/gputronic_gold_lib.o
	@echo "[GPUTronic] Built: $@"

# ---- Example demo linking the library ----
demo: $(BUILD_DIR)/gputronic_demo

$(BUILD_DIR)/gputronic_demo: examples/gold_demo.c $(BUILD_DIR)/libgputronic_gold.a include/gputronic.h | $(BUILD_DIR)
	@echo "[GPUTronic] Building gold demo..."
	$(NVCC) $(COMMON_FLAGS) -o $@ examples/gold_demo.c -L$(BUILD_DIR) -lgputronic_gold $(LIBS)
	@echo "[GPUTronic] Built: $@"

# ---- Watchable Gold closed-loop (same binary as demo) ----
watch: demo
	./$(BUILD_DIR)/gputronic_demo

# ---- Transfer plant (cooperative GEMM tiles; Gold 1.0 stays frozen) ----
transfer: $(BUILD_DIR)/gputronic_transfer

$(BUILD_DIR)/gputronic_transfer: src/gputronic_transfer.cu $(BUILD_DIR)/libgputronic_gold.a include/gputronic.h | $(BUILD_DIR)
	@echo "[GPUTronic] Building transfer plant..."
	$(NVCC) $(COMMON_FLAGS) -o $@ src/gputronic_transfer.cu -L$(BUILD_DIR) -lgputronic_gold $(LIBS)
	@echo "[GPUTronic] Built: $@"

transfer-check: transfer
	./$(BUILD_DIR)/gputronic_transfer check

# ---- Episode plant (launch-complete GEMM; Gold 1.0 stays frozen) ----
episode: $(BUILD_DIR)/gputronic_episode

$(BUILD_DIR)/gputronic_episode: src/gputronic_episode.cu $(BUILD_DIR)/libgputronic_gold.a include/gputronic.h | $(BUILD_DIR)
	@echo "[GPUTronic] Building episode plant..."
	$(NVCC) $(COMMON_FLAGS) -o $@ src/gputronic_episode.cu -L$(BUILD_DIR) -lgputronic_gold $(LIBS)
	@echo "[GPUTronic] Built: $@"

episode-check: episode
	./$(BUILD_DIR)/gputronic_episode check

episode-proof: episode
	./$(BUILD_DIR)/gputronic_episode proof

# ---- The floor: ceiling, width map, loop, release ----
show: $(BUILD_DIR)/gputronic_show
	./$(BUILD_DIR)/gputronic_show

$(BUILD_DIR)/gputronic_show: src/gputronic_show.cu $(BUILD_DIR)/libgputronic_gold.a include/gputronic.h | $(BUILD_DIR)
	@echo "[GPUTronic] Building show..."
	$(NVCC) $(COMMON_FLAGS) -o $@ src/gputronic_show.cu -L$(BUILD_DIR) -lgputronic_gold $(LIBS)
	@echo "[GPUTronic] Built: $@"

# ---- Test gates ----
check: gold
	./$(BUILD_DIR)/gputronic_gold check

dyno: gold
	./$(BUILD_DIR)/gputronic_gold dyno

step: gold
	./$(BUILD_DIR)/gputronic_gold step

run: gold
	./$(BUILD_DIR)/gputronic_gold run 30

clean:
	@echo "[GPUTronic] Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	@echo "[GPUTronic] Clean complete"

help:
	@echo "GPUTronic Gold 1.0.0"
	@echo ""
	@echo "  make gold           - build ./build/gputronic_gold"
	@echo "  make lib            - build ./build/libgputronic_gold.a"
	@echo "  make demo           - build watchable Gold closed-loop example"
	@echo "  make watch          - build and run the 30s Gold watch demo"
	@echo "  make check          - Gold dyno + closed-loop gate"
	@echo "  make transfer       - cooperative GEMM plant (Gold frozen)"
	@echo "  make transfer-check - transfer dyno + closed-loop gate"
	@echo "  make transfer && ./build/gputronic_transfer mask"
	@echo "  make episode        - launch-complete GEMM plant (Gold frozen)"
	@echo "  make show           - ceiling, width map, loop, release"
	@echo "  make episode-check  - episode dyno + closed-loop gate"
	@echo "  make episode-proof - twin / tax / open-loop hold / stock PI"
	@echo "  make dyno           - Gold sleep→rate linearity sweep"
	@echo "  make run            - 30s Gold closed-loop free-run"
	@echo "  make clean"

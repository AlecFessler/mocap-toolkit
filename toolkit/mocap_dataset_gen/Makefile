CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g -O3 -D_GLIBCXX_USE_CXX11_ABI=1

LIBTORCH_PATH = /usr/local/libtorch
CUDA_PATH = /usr/local/cuda-12.3

COMMON_DIR = ../common
COMMON_SRC_DIR = $(COMMON_DIR)/src
COMMON_INC_DIR = $(COMMON_DIR)/include
DATASET_GEN_SRC_DIR = src
DATASET_GEN_INC_DIR = include
OBJ_DIR = obj
BIN_DIR = bin
COMMON_OBJ_DIR = $(OBJ_DIR)/common
DATASET_GEN_OBJ_DIR = $(OBJ_DIR)/calib

PKG_OPENCV = $(shell pkg-config --cflags opencv4)
PKG_LIBS_OPENCV = $(shell pkg-config --libs opencv4)

INCLUDES = -I$(COMMON_INC_DIR) \
          -I$(DATASET_GEN_INC_DIR) \
          -I$(LIBTORCH_PATH)/include \
          -I$(LIBTORCH_PATH)/include/torch/csrc/api/include \
          -I$(CUDA_PATH)/include \
          $(PKG_OPENCV)

LIBS = $(PKG_LIBS_OPENCV) \
       -L$(LIBTORCH_PATH)/lib -L$(CUDA_PATH)/lib64 \
       -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
       -lcudart -lcnpy -lz -lyaml -pthread

COMMON_SRCS = $(wildcard $(COMMON_SRC_DIR)/*.cpp)
DATASET_GEN_SRCS = $(wildcard $(DATASET_GEN_SRC_DIR)/*.cpp)

COMMON_OBJS = $(COMMON_SRCS:$(COMMON_SRC_DIR)/%.cpp=$(COMMON_OBJ_DIR)/%.o)
DATASET_GEN_OBJS = $(DATASET_GEN_SRCS:$(DATASET_GEN_SRC_DIR)/%.cpp=$(DATASET_GEN_OBJ_DIR)/%.o)

$(shell mkdir -p $(BIN_DIR) $(COMMON_OBJ_DIR) $(DATASET_GEN_OBJ_DIR))

all: $(BIN_DIR)/mocap_dataset_gen

$(BIN_DIR)/mocap_dataset_gen: $(COMMON_OBJS) $(DATASET_GEN_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

$(COMMON_OBJ_DIR)/%.o: $(COMMON_SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(DATASET_GEN_OBJ_DIR)/%.o: $(DATASET_GEN_SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR)
	rm -rf $(BIN_DIR)

.PHONY: all clean

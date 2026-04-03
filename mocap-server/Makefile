CXX = g++
NVCC = /opt/cuda/bin/nvcc
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

CUDA_PATH = /opt/cuda
TENSORRT_PATH = /usr

SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = bin

INCLUDES = -I$(INC_DIR) \
           -I$(TENSORRT_PATH)/include \
           -I$(CUDA_PATH)/include \
           -I/usr/include/opencv4

PKG_AVCODEC = $(shell pkg-config --cflags libavcodec libavutil)
INCLUDES += $(PKG_AVCODEC)

PKG_LIBS_AVCODEC = $(shell pkg-config --libs libavcodec libavutil)
LIBS = -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_calib3d \
       -L$(TENSORRT_PATH)/lib -L$(CUDA_PATH)/lib64 \
       -lnvinfer -lcudart \
       -lz -lyaml -pthread -latomic \
       $(PKG_LIBS_AVCODEC)

NVCCFLAGS = -std=c++17 -O2 $(addprefix -I,$(INC_DIR) $(CUDA_PATH)/include)

CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS = $(wildcard $(SRC_DIR)/*.cu)

CPP_OBJS = $(CPP_SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
CU_OBJS = $(CU_SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
OBJS = $(CPP_OBJS) $(CU_OBJS)

$(shell mkdir -p $(BIN_DIR) $(OBJ_DIR))

all: $(BIN_DIR)/mocap-toolkit

$(BIN_DIR)/mocap-toolkit: $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

install: $(BIN_DIR)/mocap-toolkit
	@echo "Installing mocap-toolkit to /usr/local/bin/mocap-toolkit"
	@sudo install -m 755 $(BIN_DIR)/mocap-toolkit /usr/local/bin/mocap-toolkit

uninstall:
	@echo "Removing mocap-toolkit from /usr/local/bin/mocap-toolkit"
	@sudo rm -f /usr/local/bin/mocap-toolkit

.PHONY: all clean install uninstall

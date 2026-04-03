CXX = g++
NVCC = /opt/cuda/bin/nvcc
GLSLC = glslc
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

CUDA_PATH = /opt/cuda
TENSORRT_PATH = /usr

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
SHADER_DIR = src/render/shaders

MODULES = core net video model cuda calibration triangulation render

INCLUDES = -I$(SRC_DIR) \
           -I$(TENSORRT_PATH)/include \
           -I$(CUDA_PATH)/include \
           -I/usr/include/opencv4

PKG_AVCODEC = $(shell pkg-config --cflags libavcodec libavutil)
INCLUDES += $(PKG_AVCODEC)

PKG_LIBS_AVCODEC = $(shell pkg-config --libs libavcodec libavutil)
LIBS = -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_calib3d \
       -L$(TENSORRT_PATH)/lib -L$(CUDA_PATH)/lib64 \
       -lnvinfer -lcudart \
       -lvulkan -lglfw \
       -lz -lyaml -pthread -latomic \
       $(PKG_LIBS_AVCODEC)

NVCCFLAGS = -std=c++17 -O2 -I$(SRC_DIR) -I$(CUDA_PATH)/include

CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp) $(foreach mod,$(MODULES),$(wildcard $(SRC_DIR)/$(mod)/*.cpp))
CU_SRCS = $(foreach mod,$(MODULES),$(wildcard $(SRC_DIR)/$(mod)/*.cu))
VERT_SRCS = $(wildcard $(SHADER_DIR)/*.vert)
FRAG_SRCS = $(wildcard $(SHADER_DIR)/*.frag)

CPP_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))
CU_OBJS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SRCS))
SPV_OBJS = $(VERT_SRCS:%.vert=%.vert.spv) $(FRAG_SRCS:%.frag=%.frag.spv)
OBJS = $(CPP_OBJS) $(CU_OBJS)

OBJ_SUBDIRS = $(sort $(dir $(OBJS)))
$(shell mkdir -p $(BIN_DIR) $(OBJ_SUBDIRS))

all: shaders $(BIN_DIR)/mocap-toolkit

shaders: $(SPV_OBJS)

$(BIN_DIR)/mocap-toolkit: $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.vert.spv: %.vert
	$(GLSLC) $< -o $@

%.frag.spv: %.frag
	$(GLSLC) $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(SHADER_DIR)/*.spv

install: $(BIN_DIR)/mocap-toolkit
	@echo "Installing mocap-toolkit to /usr/local/bin/mocap-toolkit"
	@sudo install -m 755 $(BIN_DIR)/mocap-toolkit /usr/local/bin/mocap-toolkit

uninstall:
	@echo "Removing mocap-toolkit from /usr/local/bin/mocap-toolkit"
	@sudo rm -f /usr/local/bin/mocap-toolkit

.PHONY: all clean install uninstall shaders

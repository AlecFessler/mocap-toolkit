CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g -I/usr/include/opencv4

COMMON_DIR = ../common
COMMON_SRC_DIR = $(COMMON_DIR)/src
COMMON_INC_DIR = $(COMMON_DIR)/include

CALIB_SRC_DIR = src
CALIB_INC_DIR = include

OBJ_DIR = obj
BIN_DIR = bin

COMMON_OBJ_DIR = $(OBJ_DIR)/common
CALIB_OBJ_DIR = $(OBJ_DIR)/calib

COMMON_SRCS = $(wildcard $(COMMON_SRC_DIR)/*.cpp)
CALIB_SRCS = $(wildcard $(CALIB_SRC_DIR)/*.cpp)

COMMON_OBJS = $(COMMON_SRCS:$(COMMON_SRC_DIR)/%.cpp=$(COMMON_OBJ_DIR)/%.o)
CALIB_OBJS = $(CALIB_SRCS:$(CALIB_SRC_DIR)/%.cpp=$(CALIB_OBJ_DIR)/%.o)

LIBS = -lopencv_core -lopencv_imgproc -lrt -pthread
INCLUDES = -I$(COMMON_INC_DIR) -I$(CALIB_INC_DIR)

$(shell mkdir -p $(BIN_DIR) $(COMMON_OBJ_DIR) $(CALIB_OBJ_DIR))

all: $(BIN_DIR)/lens_calibration

$(BIN_DIR)/lens_calibration: $(COMMON_OBJS) $(CALIB_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

$(COMMON_OBJ_DIR)/%.o: $(COMMON_SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(CALIB_OBJ_DIR)/%.o: $(CALIB_SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR)
	rm -rf $(BIN_DIR)

.PHONY: all clean

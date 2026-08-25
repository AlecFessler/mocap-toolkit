CXX      = g++
CXXFLAGS = -std=c++23 -fno-exceptions -Wall -Wextra
DEPFLAGS = -MMD -MP

# -O2 on toml++'s template instantiations dominates build time, so iterate at
# -O0 and opt in to optimisation with `make release`.
OPT ?= -O0 -g
CXXFLAGS += $(OPT)

LIB_DIR = server_lib
OBJ_DIR = obj
BIN_DIR = bin
OUT_LIB = lib

LIB      = $(OUT_LIB)/libmocapserver.so
LIB_SRCS = $(wildcard $(LIB_DIR)/*.cpp)
LIB_OBJS = $(LIB_SRCS:$(LIB_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# each tool is a standalone binary linking the shared library
TOOLS = lens_calibration
BINS  = $(TOOLS:%=$(BIN_DIR)/%)

INCLUDES = -I$(LIB_DIR)
# $$ORIGIN lets the binaries find the .so relative to themselves, uninstalled
RPATH    = -Wl,-rpath,'$$ORIGIN/../$(OUT_LIB)'

DEPS = $(LIB_OBJS:.o=.d) $(TOOLS:%=$(OBJ_DIR)/%.o.d)

.PHONY: all run release clean

all: $(LIB) $(BINS)

$(LIB): $(LIB_OBJS) | $(OUT_LIB)
	$(CXX) $(CXXFLAGS) -shared $^ -o $@

$(OBJ_DIR)/%.o: $(LIB_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -fPIC $(INCLUDES) -c $< -o $@

$(BIN_DIR)/%: %/main.cpp $(LIB) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) $< -o $@ -L$(OUT_LIB) -lmocapserver $(RPATH)

$(OBJ_DIR) $(BIN_DIR) $(OUT_LIB):
	@mkdir -p $@

run: all
	@./$(BIN_DIR)/lens_calibration

release:
	@$(MAKE) --no-print-directory OPT="-O2 -g" clean all

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(OUT_LIB)

-include $(DEPS)

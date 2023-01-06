CC=mpic++

BIN_DIR = bin
OBJ_DIR = obj
SRC_DIR = src

# CMD variables with their default values
BUILD_TYPE ?= release
BUILD_ENV ?= local

# Set the appropriate compile/link flags
CFLAGS = -Wall -std=c++17
LFLAGS = -Wall -std=c++17

## Add flags based on BUILD_TYPE (release or debug)
ifeq ($(BUILD_TYPE), release)
	CFLAGS += -O3
	LFLAGS += -O3
else ifeq ($(BUILD_TYPE), debug)
	BIN_DIR := ${BIN_DIR}_debug
	OBJ_DIR := ${OBJ_DIR}_debug
	CFLAGS += -g -O0 -DDEBUG
	LFLAGS += -g -O0 -DDEBUG
endif

## Add flags based on BUILD_ENV (local or hpc)
ifeq ($(BUILD_ENV), local)
	CFLAGS += -fopenmp -lblas
	LFLAGS += -fopenmp -lblas
else ifeq ($(BUILD_ENV), hpc)
	CFLAGS += -fopenmp $$(pkgconf --cflags openblas)
	LFLAGS += -fopenmp $$(pkgconf --cflags -libs openblas)
endif


# Initiate all variables needed for building
EXEC_CPP = $(wildcard $(SRC_DIR)/*.cpp)
EXEC_OBJ = $(EXEC_CPP:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
EXEC_BIN = $(EXEC_CPP:$(SRC_DIR)/%.cpp=$(BIN_DIR)/%)

COMMON_CPP = $(wildcard $(SRC_DIR)/detail/*.cpp)
COMMON_OBJ = $(COMMON_CPP:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

.SECONDARY: $(COMMON_OBJ) $(EXEC_OBJ) #Added this so that .o files aren't deleted

DEPS = $(wildcard $(SRC_DIR)/detail/*.hpp)


# Actual build target
do_build: $(info $$BIN_DIR is [${BIN_DIR}])$(EXEC_BIN)



$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(DEPS)
	@mkdir -p '$(@D)'
	$(CC) -c $< -o $@ $(CFLAGS)

$(BIN_DIR)/%: $(OBJ_DIR)/%.o $(COMMON_OBJ)
	@mkdir -p '$(@D)'
	$(CC) -o $@ $^ $(LFLAGS)



.PHONY: clean
clean:
	rm -rf $(BIN_DIR)/* $(OBJ_DIR)/*

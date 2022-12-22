# The mpi compiler
CC = mpic++

# The name of the executable file to be created
EXE = my_program

# Compiler flags
CFLAGS = -c -Wall -O3

# The directories where object files and executables will be created
OBJ_DIR = obj
BIN_DIR = bin

# The directory where source files are located
SRC_DIR = src

# List of all source files
SRCS = $(wildcard $(SRC_DIR)/*.cpp)

# List of all object files
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Link object files
$(BIN_DIR)/$(EXE): $(OBJS)
	$(CC) $^ $(LDFLAGS) -o $@

# Compile
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) $< -o $@

# Builds the default executable
all: mkdir $(BIN_DIR)/$(EXE)

.PHONY: clean
clean:
	rm -f $(BIN_DIR)/$(EXE) $(OBJS)

# Create directories
.PHONY: mkdir
mkdir:
	mkdir -p $(OBJ_DIR)
	mkdir -p $(BIN_DIR)
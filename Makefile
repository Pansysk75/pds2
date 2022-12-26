# The mpi compiler
CC = mpic++

# Compiler flags
CFLAGS = -Wall -O3

# The directories where object files and executables will be created
BIN_DIR = bin

# The directory where source files are located
SRC_DIR = src


# Builds the default executable
all: mkdir mpi


# Compile
mpi: $(SRC_DIR)/*
	$(CC) $(CFLAGS) $(SRC_DIR)/$@.cpp -o $(BIN_DIR)/$@


# Create directories
.PHONY: mkdir
mkdir:
	mkdir -p $(BIN_DIR)
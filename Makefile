CC=mpic++

BIN_DIR = bin/
OBJ_DIR = obj/
SRC_DIR = src/


DEBUG ?= 0
ifeq ($(DEBUG), 1)
#		CFLAGS = -Wall -std=c++17 -g -O0 -DDEBUG -fopenmp $$(pkgconf --cflags openblas)
#       LFLAGS = -Wall -std=c++17 -g -O0 -DDEBUG -fopenmp $$(pkgconf --cflags -libs openblas)
    CFLAGS = -Wall -std=c++17 -lblas -g -O0 -DDEBUG -fopenmp
	LFLAGS = -Wall -std=c++17 -lblas -g -O0 -DDEBUG -fopenmp
	OBJ_DIR = obj_debug/
	BIN_DIR = bin_debug/

else
#		CFLAGS = -Wall -std=c++17 -O3 -fopenmp $$(pkgconf --cflags openblas)
#       LFLAGS = -Wall -std=c++17 -O3 -fopenmp $$(pkgconf --cflags -libs openblas)
    CFLAGS = -Wall -std=c++17 -O3 -lblas -fopenmp
	LFLAGS = -Wall -std=c++17 -O3 -lblas -fopenmp
	OBJ_DIR = obj/
	BIN_DIR = bin/
endif


EXEC_CPP = $(wildcard $(SRC_DIR)*.cpp)
EXEC_OBJ = $(EXEC_CPP:$(SRC_DIR)%.cpp=$(OBJ_DIR)%.o)
EXEC_BIN = $(EXEC_CPP:$(SRC_DIR)%.cpp=$(BIN_DIR)%)

COMMON_CPP = $(wildcard $(SRC_DIR)detail/*.cpp)
COMMON_OBJ = $(COMMON_CPP:$(SRC_DIR)%.cpp=$(OBJ_DIR)%.o)

.SECONDARY: $(COMMON_OBJ) $(EXEC_OBJ) #Added this so that .o files aren't deleted

DEPS = $(wildcard $(SRC_DIR)detail/*.hpp)

all: $(EXEC_BIN)
release: $(EXEC_BIN)
debug: $(EXEC_BIN)



$(OBJ_DIR)%.o: $(SRC_DIR)%.cpp $(DEPS)
	@mkdir -p '$(@D)'
	$(CC) -c $< -o $@ $(CFLAGS)

$(BIN_DIR)%: $(OBJ_DIR)%.o $(COMMON_OBJ)
	@mkdir -p '$(@D)'
	$(CC) -o $@ $^ $(LFLAGS)



.PHONY: clean
clean:
	rm -rf $(BIN_DIR)* $(OBJ_DIR)*

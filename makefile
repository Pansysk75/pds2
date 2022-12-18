CC=g++

CFLAGS = -Wall -g -O3 -std=c++20
LFLAGS = -Wall -g -O3 -std=c++20
LIBS = -lblas

DEPS = knn.hpp fileio.hpp testingknn.hpp global_vars.hpp
OBJ = fileio.o knn.o testingknn.o main.o global_vars.o

%.o: %.cpp $(DEPS)
	$(CC) -c $< -o $@ $(CFLAGS) $(LIBS)

knn: $(OBJ)
	$(CC) -o $@ $^ $(LFLAGS) $(LIBS)

.PHONY: clean
clean:
	rm -f *.o knn

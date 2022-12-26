CC=g++

OPT = -O3
DEB = -g 

CFLAGS = -Wall $(DEB) $(OPT) -std=c++20
LFLAGS = -Wall $(DEB) $(OPT) -std=c++20
LIBS = -lblas

DEPS = knnDist.hpp fileio.hpp testingknn.hpp global_includes.hpp
OBJ = fileio.o knnDist.o testingknn.o main.o global_includes.o

%.o: %.cpp $(DEPS)
	$(CC) -c $< -o $@ $(CFLAGS) $(LIBS)

knn: $(OBJ)
	$(CC) -o $@ $^ $(LFLAGS) $(LIBS)

.PHONY: clean
clean:
	rm -f *.o knn

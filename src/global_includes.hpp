# pragma once
#include <iostream>

#define DEB(msg) if(debug) std::cout << msg << std::endl;
#define idx(i, j, ld) (((i)*(ld))+(j))
extern bool magic;
extern bool debug;
extern size_t max_batch_size;
extern size_t num_batches;
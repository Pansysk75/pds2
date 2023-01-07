#pragma once

#include "communication.hpp"
#include "knn_structs.hpp"
#include "knn_utils.hpp"
#include "knn_algorithms.hpp"
#include "fileio.hpp"
#include "utilities.hpp"

#define MASTER_RANK 0

struct initial_work_data
{
    // Must be passed to every worker proccess, stores
    // data that is not known at compile time
    std::vector<char> filename;
    size_t idx_start;
    size_t idx_end;
    size_t max_size; // needed for safe memory allocation
    size_t d; // dimensionality of point-space
    size_t k; // number of nearest neighbours that should be found

    initial_work_data(std::string filename, size_t idx_start, size_t idx_end, size_t max_size, size_t d, size_t k);

    initial_work_data(std::vector<char> filename, size_t idx_start, size_t idx_end, size_t max_size, size_t d, size_t k);
};

// Describe how to com_port::send this class

template <>
inline void com_port::_impl_send(int destination_id, initial_work_data &d)
{
    send(destination_id, d.filename, d.idx_start, d.idx_end, d.max_size, d.d, d.k);
}

// Describe how to com_port::receive this class

template <>
inline void com_port::_impl_receive(int source_id, initial_work_data &d)
{
    receive(source_id, d.filename, d.idx_start, d.idx_end, d.max_size, d.d, d.k);
}


// Worker receives inital data from the master mpi process.
// Then, it loads its assigned part of a data file.
// Then, the workers will process and exchange data in a cyclical pattern, 
// until all workers have processed all data.
// The results are then sent back to the master process.
class worker
{
private:
    utilities::tracer tracer;

public:
    // Communication port which facilitates all communication
    com_port com;

    // Initial data which will be received from the "master" process
    initial_work_data init_data;

    // Points this worker is responsible for (constant for the lifetime of this worker)
    QueryPacket query;

    // Points which are processed in one iteration (different for every iteration)
    CorpusPacket corpus;

    // A set that is allocated for receiving a CorpusPacket.
    CorpusPacket receiving_corpus;

    // KNN result for local points, will be combined in master process
    ResultPacket results;


    worker(int rank, int world_size);
    worker(int rank, int world_size, initial_work_data init_data);

    void initialize();

    void print_debug();
    void print_debug(std::string str);

    void work();
};

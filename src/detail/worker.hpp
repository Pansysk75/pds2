#pragma once
#include "communication.hpp"
#include "knn_structs.hpp"
#include "knn_utils.hpp"
#include "knn_algorithms.hpp"
#include "fileio.hpp"

#define MASTER_RANK 0

struct initial_work_data
{
    // Must be passed to every worker proccess, stores
    // data that is not known at compile time
    size_t n; // number of d-dimensional points in each packet
    size_t d; // dimensionality of point-space
    size_t k; // number of nearest neighbours that should be found
};

template <>
void com_port::_impl_send(int destination_id, initial_work_data &d)
{
    send(destination_id, d.n, d.d, d.k);
}
template <>
void com_port::_impl_receive(int source_id, initial_work_data &d)
{
    receive(source_id, d.n, d.d, d.k);
}

class worker
{

public:
    // Communication port which facilitates all communication
    com_port com;

    // Initial data which will be received from a "master" process
    initial_work_data init_data;

    // Points this worker is responsible for (constant for the lifetime of this worker)
    QueryPacket query;

    // Points which are processed in one iteration (different for every iteration)
    CorpusPacket corpus;

    // A set that is allocated for receiving a points_chunk.
    CorpusPacket receiving_corpus;

    // KNN result for local points, will be combined in master process
    ResultPacket results;

    worker(int rank, int world_size)
        : com(rank, world_size)
    {
        // Initialize worker and receive inital data / corpus set
        // The master mpi process is responsible for transfering initial
        // data to all workers. Then, the workers will exchange data in a
        // cyclical pattern, until all workers have processed all data.
        com.receive(MASTER_RANK, init_data);
        init();
    }

    worker(int rank, int world_size, initial_work_data init_data)
        : com(rank, world_size), init_data(init_data)
    {
        init();
    }

    void init()
    {
        int size = init_data.n;
        int dim = init_data.d;

        int idx_start = com.rank() * size;
        int idx_end = (com.rank() + 1) * size;

        query = QueryPacket(size, dim, idx_start, idx_end);
        query.X = import_data(idx_start, idx_end, dim);

        corpus = CorpusPacket(size, dim, idx_start, idx_end);
        corpus.Y = import_data(idx_start, idx_end, dim);

        receiving_corpus = CorpusPacket(0, 0, 0, 0);
        receiving_corpus.Y.resize(size * dim);

        // results = ResultPacket(size, 0, init_data.k, idx_start, idx_end, idx_start, idx_end);

        results = ResultPacket(0, 0, 0, 0, 0, 0, 0);

        deb("Initialization complete!");
    }

    void deb(std::string str)
    {
        std::cout << com.rank() << ": " << str << std::endl;
    }

    void deb_v()
    {
        std::string deb_str;
        deb_str += "query : ";
        deb_str += std::to_string(query.d) + " | ";
        deb_str += std::to_string(query.m_packet) + " | ";
        deb_str += std::to_string(query.x_start_index) + "->";
        deb_str += std::to_string(query.x_end_index) + " | ";
        // for (auto& elem : query.X) deb_str += std::to_string(elem) += " ";

        deb_str += "\n   ";

        deb_str += "corpus: ";
        deb_str += std::to_string(corpus.d) + " | ";
        deb_str += std::to_string(corpus.n_packet) + " | ";
        deb_str += std::to_string(corpus.y_start_index) + "->";
        deb_str += std::to_string(corpus.y_end_index) + " | ";
        // for (auto& elem : corpus.Y) deb_str += std::to_string(elem) += " ";

        deb_str += "\n   ";

        deb_str += "res: " + std::to_string(results.x_start_index) + "->" + std::to_string(results.x_end_index) + " | " + std::to_string(results.y_start_index) + "->" + std::to_string(results.y_end_index) + " | m:" + std::to_string(results.m_packet) + " n:" + std::to_string(results.n_packet) + " | k:" + std::to_string(results.k);

        deb(deb_str);
    }

    void work()
    {

        for (int i = 0; i < com.world_size() - 1; i++)
        {

            // Debug s*&.. stuff
            deb("Started iteration " + std::to_string(i));

            int next_rank = (com.rank() + 1) % com.world_size();
            int prev_rank = (com.rank() + com.world_size() - 1) % com.world_size();

            // Start sending the part we just proccessed
            // Start receiving the part we will proccess later
            com_request send_req = com.send_begin(next_rank, corpus);
            com_request recv_req = com.receive_begin(prev_rank, receiving_corpus);
            // com.send(corpus, next_rank);
            // com.receive(receiving_corpus, prev_rank);

            // Work on working set
            ResultPacket batch_result = knn_blas(query, corpus, init_data.k);
            // Combine this result with previous results
            if (results.n_packet == 0)
                results = std::move(batch_result);
            else
                results = combineKnnResultsSameX(batch_result, results);

            // debug worker state
            deb_v();

            // Wait for open communications to finish
            com.wait(send_req);
            com.wait(recv_req);

            // Update query_set with received set (using std::swap is the
            // equivelant of swapping the pointers of two C arrays)
            std::swap(corpus, receiving_corpus);
        }
        // Work on last batch
        ResultPacket batch_result = knn_blas(query, corpus, init_data.k);
        // Combine this result with previous results
        results = combineKnnResultsSameX(batch_result, results);
        deb_v();

        MPI_Barrier(MPI_COMM_WORLD);
        // Work finished, send results to master process
        if (com.rank() != MASTER_RANK)
        {
            com.send(MASTER_RANK, results);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
};

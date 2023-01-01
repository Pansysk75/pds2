#pragma once
#include "communication.hpp"
#include "knn_structs.hpp"
#include "knn_utils.hpp"
#include "knn_algorithms.hpp"
#include "fileio.hpp"
#include "testingknn.hpp"

#define MASTER_RANK 0

struct initial_work_data
{
    // Must be passed to every worker proccess, stores
    // data that is not known at compile time
    std::vector<char> filename;
    size_t idx_start;
    size_t idx_end;
    size_t max_size;
    size_t d; // dimensionality of point-space
    size_t k; // number of nearest neighbours that should be found

    initial_work_data(std::string filename, size_t idx_start, size_t idx_end, size_t max_size, size_t d, size_t k)
    :filename(filename.begin(), filename.end()),
    idx_start(idx_start), idx_end(idx_end),max_size(max_size), d(d),k(k){}

    initial_work_data(std::vector<char> filename, size_t idx_start, size_t idx_end, size_t max_size, size_t d, size_t k)
    :filename(filename), 
    idx_start(idx_start),idx_end(idx_end), max_size(max_size), d(d),k(k){}
};

template <>
void com_port::_impl_send(int destination_id, initial_work_data &d)
{
    send(destination_id, d.filename, d.idx_start, d.idx_end, d.max_size, d.d, d.k);
}
template <>
void com_port::_impl_receive(int source_id, initial_work_data &d)
{
    receive(source_id, d.filename, d.idx_start, d.idx_end, d.max_size, d.d, d.k);
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
        : com(rank, world_size),
        init_data(std::vector<char>(128),0,0,0,0,0)
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
        int max_size = init_data.max_size;
        int d = init_data.d;

        int start_idx = init_data.idx_start;
        int end_idx = init_data.idx_end;

        std::string filename(init_data.filename.begin(), init_data.filename.end());

        deb(filename + " " + std::to_string(start_idx) + " " + std::to_string(end_idx));

        auto [temp_query, temp_corpus] = file_packets(filename, start_idx, end_idx, d);
        query = std::move(temp_query);
        corpus = std::move(temp_corpus);

        // query.X.resize(max_size*d);
        corpus.Y.resize(max_size*d);
        receiving_corpus = CorpusPacket(max_size, d, 0, 0);

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
            deb_v();

            int next_rank = (com.rank() + 1) % com.world_size();
            int prev_rank = (com.rank() + com.world_size() - 1) % com.world_size();

            // Start sending the part we just proccessed
            // Start receiving the part we will proccess later
            com_request send_req = com.send_begin(prev_rank, corpus);
            com_request recv_req = com.receive_begin(next_rank, receiving_corpus);

            // Work on working set
            ResultPacket batch_result = knn_blas(query, corpus, init_data.k);
            // Combine this result with previous results
    
   
            results = combineKnnResultsSameX(results, batch_result);
            

            // debug worker state
            deb_v();

            // Wait for open communications to finish
            com.wait(send_req);
            com.wait(recv_req);

            deb("Finished transmission #" + std::to_string(i));

            // Update query_set with received set (using std::swap is the
            // equivelant of swapping the pointers of two C arrays)
            std::swap(corpus, receiving_corpus);
        }
        // Work on last batch
        ResultPacket batch_result = knn_blas(query, corpus, init_data.k);
        // Combine this result with previous results
        results = combineKnnResultsSameX(results, batch_result);
        deb_v();

        // Work finished, send results to master process
        if (com.rank() != MASTER_RANK)
        {
            com.send(MASTER_RANK, results);
        }

    }
};

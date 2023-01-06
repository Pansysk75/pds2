#include "communication.hpp"
#include "knn_structs.hpp"
#include "knn_utils.hpp"
#include "knn_algorithms.hpp"
#include "fileio.hpp"
#include "worker.hpp"

#include "globals.hpp"

initial_work_data::initial_work_data(std::string filename, size_t idx_start, size_t idx_end, size_t max_size, size_t d, size_t k)
    : filename(filename.begin(), filename.end()),
      idx_start(idx_start), idx_end(idx_end), max_size(max_size), d(d), k(k) {}

initial_work_data::initial_work_data(std::vector<char> filename, size_t idx_start, size_t idx_end, size_t max_size, size_t d, size_t k)
    : filename(filename),
      idx_start(idx_start), idx_end(idx_end), max_size(max_size), d(d), k(k) {}

worker::worker(int rank, int world_size)
    : com(rank, world_size),
      init_data(std::vector<char>(128), 0, 0, 0, 0, 0)
{
    com.receive(MASTER_RANK, init_data);
    initialize();
}

worker::worker(int rank, int world_size, initial_work_data init_data)
    : com(rank, world_size), init_data(init_data)
{
    initialize();
}

void worker::initialize()
{
    int max_size = init_data.max_size;
    int d = init_data.d;

    int start_idx = init_data.idx_start;
    int end_idx = init_data.idx_end;

    std::string filename(init_data.filename.begin(), init_data.filename.end());

    print_debug("Importing data from " + filename + ": " + std::to_string(start_idx) + "->" + std::to_string(end_idx));

    auto [temp_query, temp_corpus] = file_packets(filename, start_idx, end_idx, d);
    query = std::move(temp_query);
    corpus = std::move(temp_corpus);

    // query.X.resize(max_size*d);
    corpus.Y.resize(max_size * d);
    receiving_corpus = CorpusPacket(max_size, d, 0, 0);

    results = ResultPacket(0, 0, 0, 0, 0, 0, 0);

    print_debug("Initialization complete!");
}

void worker::print_debug()
{
    if(globals::debug) std::cout << "\n"
              << com.rank() << ": ";
    if(globals::debug) std::cout << "\tquery: " << query;
    if(globals::debug) std::cout << "\n\tcorpus: " << corpus;
    if(globals::debug) std::cout << "\n\tresult: " << results << std::endl;
}

void worker::print_debug(std::string str)
{
    if(globals::debug) std::cout << "\n"
              << com.rank() << ": " << str << std::endl;
}

void worker::work()
{

    for (int i = 0; i < com.world_size() - 1; i++)
    {

        print_debug("Started iteration " + std::to_string(i));
        print_debug();

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
        print_debug();

        // Wait for open communications to finish
        com.wait(send_req);
        com.wait(recv_req);

        print_debug("Finished transmission #" + std::to_string(i));

        // Update query_set with received set (using std::swap is the
        // equivelant of swapping the pointers of two C arrays)
        std::swap(corpus, receiving_corpus);
    }
    // Work on last batch
    ResultPacket batch_result = knn_blas(query, corpus, init_data.k);
    // Combine this result with previous results
    results = combineKnnResultsSameX(results, batch_result);
    print_debug();

    // Work finished, send results to master process
    if (com.rank() != MASTER_RANK)
    {
        com.send(MASTER_RANK, results);
    }
}

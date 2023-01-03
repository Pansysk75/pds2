#include <iostream>
#include <string>
#include <vector>

#include "detail/communication.hpp"
#include "detail/knn_structs.hpp"
#include "detail/mpi_process.hpp"
#include "detail/utilities.hpp"
#include "detail/worker.hpp"

#define MASTER_RANK 0


void print_results(const QueryPacket &query, const CorpusPacket &corpus,
                   const ResultPacket &result, size_t k)
{
    // Print the results
    for (size_t i = 0; i < std::min(result.m_packet, 10ul); i++)
    {
        std::cout << "The calculated " << k << " nearest neighbours of number: "
                  << query.X[idx(i, 0, query.d)] << std::endl;
        std::cout << "Are actually the numbers:" << std::endl;

        for (size_t j = 0; j < std::min(result.k, (size_t)5ul); j++)
        {
            std::cout << corpus.Y[idx(result.nidx[idx(i, j, result.k)], 0, corpus.d)]
                      << " with an MSE of: " << result.ndist[idx(i, j, result.k)]
                      << std::endl;
        }
        std::cout << std::endl;
    }
}

// Entry point for MPI master
ResultPacket master_main(mpi_process &process, std::string filename, size_t n_total, size_t d, size_t k)
{
    com_port com(process.world_rank, process.world_size);

    // Send initial data to all workers

    std::vector<initial_work_data> init_data;
    size_t max_size = 1 + n_total / process.world_size; //all workers will alocate memory this size of memory 

    for (size_t i = 0; i < (size_t)process.world_size; i++)
    {
        size_t idx_start = (i * n_total) / process.world_size; // will this overflow?
        size_t idx_end = ((i + 1) * n_total) / process.world_size;

        init_data.push_back(
            initial_work_data(filename, idx_start, idx_end, max_size, d, k));

        std::cout << "Worker " << i << ": " << filename << " n=" << idx_start << "->" << idx_end << std::endl;

        // send to everyone except this
        if (i != MASTER_RANK)
            com.send(i, init_data[i]);
    }
    // Initialize local worker
    worker w(process.world_rank, process.world_size, init_data[0]);

    w.work();

    // Now every process is working and exchanging data!
    // Time passes ...
    // ....

    // Gather result
    std::vector<ResultPacket> diffProcRes;
    diffProcRes.push_back(w.results);
    for (int i = 1; i < process.world_size; i++)
    {
        size_t query_size = (init_data[i].idx_end - init_data[i].idx_start);
        ResultPacket result(query_size * k);
        com.receive(i, result);
        diffProcRes.push_back(result);
    }


    // Print results 
    std::cout << "RESULTS\n";
    for (auto &elem : diffProcRes)
    {
        std::cout << elem << std::endl;
    }

    return combineCompleteQueries(diffProcRes);
}

void slave_main(mpi_process &process)
{
    worker w(process.world_rank, process.world_size);
    w.work();
}

// MPI entry:
int main(int argc, char **argv)
{

    mpi_process process(&argc, &argv);

    // Handle command line arguments
    if (argc != 5)
    {
        if (process.is_master()){
            std::cout << "Usage: " << argv[0] << " <filename> <n> <d> <k>"
                        << std::endl;
        }
        return 1;
    }

    
    if (process.is_master())
    {

        std::string filename = argv[1];
        size_t n_total = std::stoi(argv[2]);
        size_t d = std::stoi(argv[3]);
        size_t k = std::stoi(argv[4]);

        std::cout << "filename: " << filename << std::endl;
        std::cout << "n_total: " << n_total << std::endl;
        std::cout << "d: " << d << std::endl;
        std::cout << "k: " << k << std::endl;


        utilities::timer my_timer;
        my_timer.start();
        ResultPacket final_result = master_main(process, filename, n_total, d, k);
        my_timer.stop();

        std::cout << "FINAL RESULTS\n";
        std::cout << final_result << std::endl;

        auto [query, corpus] = file_packets(filename, 0, n_total, d);
        std::cout << "Loaded query and corpus to print" << std::endl;
        print_results(query, corpus, final_result, k);

        std::cout << "Total time: " << my_timer.get() / 1000000.0 << " ms"
                  << std::endl;
    }
    else
    {
        slave_main(process);
    }

    return 0;
}
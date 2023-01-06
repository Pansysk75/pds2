#include <iostream>
#include <string>
#include <vector>

#include "detail/communication.hpp"
#include "detail/knn_structs.hpp"
#include "detail/mpi_process.hpp"
#include "detail/utilities.hpp"
#include "detail/worker.hpp"
#include "detail/globals.hpp"

#include <unistd.h>

bool mnistPrint = false;

#define MASTER_RANK 0

void print_results_with_labels(const std::string& q_filename, const std::string& c_filename, const ResultPacket& result, size_t d)
{
    auto [query, query_labels, corpus, corpus_labels] = file_packets_with_label(q_filename, 0, result.m_packet, c_filename, 0, result.n_packet, d);

    // Print the results
    for (size_t i = 0; i < std::min(result.m_packet, 10ul); i++)
    {
        std::cout << "The calculated " << std::min(result.k, (size_t)5ul) << " nearest neighbours of vector with label: "
                  << query_labels[i] << std::endl;
        if(mnistPrint) print_mnist(&query.X[idx(i, 0, d)]);
        std::cout << "Are the ones with labels:" << std::endl;

        for (size_t j = 0; j < std::min(result.k, (size_t)5ul); j++)
        {
            std::cout << corpus_labels[result.nidx[idx(i, j, result.k)]]
                      << " with an MSE of: " << result.ndist[idx(i, j, result.k)]
                      << std::endl;
            if (mnistPrint) print_mnist(&corpus.Y[idx(result.nidx[idx(i, j, result.k)], 0, d)]);
        }
        std::cout << std::endl;
    }
}

void print_results_with_labels(const std::string& filename, const ResultPacket &result, size_t d)
{
    print_results_with_labels(filename, filename, result, d);
}

void print_results(const std::string& q_filename, const std::string& c_filename, const ResultPacket& result, size_t d)
{
    auto [query, corpus] = file_packets(q_filename, 0, result.m_packet, c_filename, 0, result.n_packet, d);

    // Print the results
    for (size_t i = 0; i < std::min(result.m_packet, 10ul); i++)
    {
        std::cout << "The calculated " << std::min(result.k, (size_t)5ul) << " nearest neighbours of point in line: "
                  << i << std::endl;
        std::cout << "Are the points in lines:" << std::endl;

        for (size_t j = 0; j < std::min(result.k, (size_t)5ul); j++)
        {
            std::cout << corpus.Y[result.nidx[idx(i, j, result.k)]]
                      << " with an MSE of: " << result.ndist[idx(i, j, result.k)]
                      << std::endl;
        }
        std::cout << std::endl;
    }
}

void print_results(const std::string& filename, const ResultPacket &result, const size_t d)
{
    print_results(filename, filename, result, d);
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

    return combineCompleteQueries(diffProcRes);
}

void slave_main(mpi_process &process)
{
    worker w(process.world_rank, process.world_size);
    w.work();
}

void printUsage(){
    std::cout << "Usage: ./mpiKnn -f=[filename] -n=[n] -d=[d] -k=[k]\noptional arguments: -p=[parts to split data in each proccess] -l if file has labels -m to enable specialized mnist printing" << std::endl;
}
// MPI entry:
int main(int argc, char **argv)
{
    mpi_process process(&argc, &argv);

    // Handle command line arguments
    if (argc < 5)
    {
        if (process.is_master()){
            printUsage();
        }
        return 1;
    }

    std::string filename = "";
    size_t n_total=0, d=0, k=0;
    
    int opt;
    while((opt = getopt(argc, argv, "f:n:d:k:p:l")) != -1){
        if(optarg){
            if(optarg[0] == '='){
                optarg = optarg + 1;
            }
        }
        switch(opt){
            case 'f':
                filename = std::string(optarg);
                break;
            case 'n':
                n_total = atoi(optarg);
                break;
            case 'd':
                d = atoi(optarg);
                break;
            case 'k':
                k = atoi(optarg);
                break;
            case 'l':
                globals::pad = true;
                break;
            case 'm':
                mnistPrint = true;
                break;
            case 'p':
                globals::parts = atoi(optarg);
                break;
            default:
                if (process.is_master()){
                    printUsage();
                }
                std::cout << "Invalid argument: " << opt << std::endl;
                return 1;
        }
    }

    if(n_total*d*k == 0 || filename == ""){
        if (process.is_master()){
            printUsage();
        }
        return 1;
    }

    if (process.is_master())
    {
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

        std::cout << "Loaded query and corpus to print" << std::endl;
        if(globals::pad){
            print_results_with_labels(filename, final_result, d);
        }else{
            print_results(filename, final_result, d);
        }

        std::cout << "Total time: " << my_timer.get() / 1000000.0 << " ms"
                  << std::endl;
    }
    else
    {
        slave_main(process);
    }

    return 0;
}
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

#define MASTER_RANK 0

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

        if(globals::debug) std::cout << "Worker " << i << ": " << filename << " n=" << idx_start << "->" << idx_end << std::endl;

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
    std::cout << "Usage: ./mpiKnn -f=[filename] -n=[n] -d=[d] -k=[k]\n\
optional arguments:\n-M=[approximate maximum memmory usage in *megabytes* for each process]\n-l if file has labels\n-P to print the results in the\n-m to enable specialized mnist printing\n-D to print out debug information" << std::endl;
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
    bool mnistPrint = false, printRes = false;
    
    int opt;
    while((opt = getopt(argc, argv, "f:n:d:k:M:lPD")) != -1){
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
            case 'D':
                globals::debug = true;
                break;
            case 'M':
                // size in megabytes
                globals::knn_part_bytes_limit = atoi(optarg)*1000000;
                break;
            case 'P':
                printRes = true;
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
        std::cout << "Running with parameters: " << std::endl;
        std::cout << "filename: " << filename << std::endl;
        std::cout << "n: " << n_total << std::endl;
        std::cout << "d: " << d << std::endl;
        std::cout << "k: " << k << std::endl;

        utilities::timer main_timer;
        main_timer.start();
        ResultPacket final_result = master_main(process, filename, n_total, d, k);
        main_timer.stop();

        std::cout << "Finished Calculations" << std::endl;
        std::cout << "X, Y ranges covered: " << final_result << std::endl;

        if(printRes){
            std::cout << "Loaded query and corpus to print" << std::endl;
            if(globals::pad){
                print_results_with_labels(filename, final_result, d, mnistPrint);
            }else{
                print_results(filename, final_result, d);
            }
        }
        std::cout << "Total time: " << main_timer.get() / 1000000.0 << " ms"
                  << std::endl;
    }
    else
    {
        slave_main(process);
    }

    return 0;
}
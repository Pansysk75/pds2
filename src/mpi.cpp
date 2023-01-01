#include <iostream>
#include <string>
#include <vector>

#include "detail/mpi_process.hpp"
#include "detail/communication.hpp"
#include "detail/knn_structs.hpp"
#include "detail/worker.hpp"
#include "detail/utilities.hpp"

#define MASTER_RANK 0

// Entry point for MPI master
void master_main(mpi_process &process)
{
    //hard coded for now
    std::string filename = "datasets/mnist_test.csv";
    size_t n_total = 5000; // # total points
    size_t d = 20;         // # dimensions
    size_t k = 10;          // # nearest neighbours

    com_port com(process.world_rank, process.world_size);


    std::vector<initial_work_data> init_data;
    size_t max_size = 1 + n_total/process.world_size;
    // Send initial data to all workers
    for (size_t i = 0; i < (size_t)process.world_size; i++)
    {
        size_t idx_start = (i * n_total) / process.world_size; // will this overflow?
        size_t idx_end = ((i+1) * n_total) / process.world_size;
        
        init_data.push_back(initial_work_data(filename, idx_start, idx_end, max_size, d, k));

        std::cout << filename << " " << idx_start << " " << idx_end << std::endl;

        // send to everyone except this
        if(i != MASTER_RANK) com.send(i, init_data[i]);
    }
    // Initialize local worker
    worker w(process.world_rank, process.world_size, init_data[0]);

    w.work();

    // Gather result
    std::vector<ResultPacket> diffProcRes;
    diffProcRes.push_back(w.results);
    for (int i = 1; i < process.world_size; i++)
    {
        std::cout << process.world_rank << "GotResult" << std::endl;
        size_t query_size = (init_data[i].idx_end - init_data[i].idx_start);
        ResultPacket result(query_size * k);
        com.receive(i, result);
        diffProcRes.push_back(result);
    }

    std::cout << "RESULTS\n";
    for (auto &elem : diffProcRes)
    {
        std::cout << elem.x_start_index << " " << elem.x_end_index << " | " << elem.y_start_index << " " << elem.y_end_index << " | " << elem.m_packet << " " << elem.n_packet << " | " << elem.k << std::endl;
    }

    ResultPacket final_result = combineCompleteQueries(diffProcRes);
}

void slave_main(mpi_process &process)
{
    worker w(process.world_rank, process.world_size);
    w.work();
}

// MPI entry:
int main()
{

    mpi_process process;

    if (process.world_rank == MASTER_RANK)
    {
        utilities::timer my_timer;
        my_timer.start();
        master_main(process);
        my_timer.stop();
        std::cout << "Total time: " << my_timer.get() / 1000000.0 << " ms" << std::endl;
    }
    else
    {

        slave_main(process);
    }

    return 0;
}
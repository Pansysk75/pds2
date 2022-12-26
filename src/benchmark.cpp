#include <iostream>
#include <string>
#include <vector>

#include <mpi/mpi.h>

#include "detail/misc.hpp"

#include "detail/mpi_process.hpp"
#include "detail/communication.hpp"
#include "detail/points_chunk.hpp"
#include "detail/worker.hpp"

double benchmark_blocking(mpi_process& process, unsigned int data_size,  unsigned int n_pieces){
    // This benchmark is to test the overhead of sending multiple packets vs sending all data in a single request
    misc::timer timer;    
    com_port com(process.world_rank, process.world_size);
    std::vector<double> vec(data_size);
    if(process.world_rank == 0){
        // Create data parcel and send
        std::iota(vec.begin(), vec.end(), 0);
        timer.start();
        for(unsigned int i=0; i<n_pieces; i++){
            unsigned int part_begin = data_size * i / n_pieces;
            unsigned int part_end = data_size * (i+1) / n_pieces;
            unsigned int part_size = part_end-part_begin;
            MPI_Send(vec.data()+part_begin, part_size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        }
        timer.stop();
        return timer.get()/1000000;

    }else if(process.world_rank == 1){
        timer.start();
        for(unsigned int i=0; i<n_pieces; i++){
            int part_begin = data_size * i / n_pieces;
            int part_end = data_size * (i+1) / n_pieces;
            int part_size = part_end-part_begin;
            MPI_Recv(vec.data()+part_begin, part_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, nullptr);
        }
        timer.stop();
        return timer.get()/1000000;
    }
    return -1;
}


// MPI entry:
int main(){

    mpi_process process;

    unsigned int data_size = 31250000/2;
    // 31250000 doubles = 250Mb

    benchmark_blocking(process, 31250000, 1);
    benchmark_blocking(process, 31250000, 1);

    unsigned int n_pieces = 1;
    for(unsigned int i=0; i<8; i++){
        double result = benchmark_blocking(process, data_size, n_pieces);

        if(process.world_rank == 0){
            std::cout << n_pieces << " pieces: "  << result << " ms" <<std::endl;
        }

        n_pieces *= 2;
    }
    


    return 0;
}
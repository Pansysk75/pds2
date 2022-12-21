#include <iostream>
#include <string>

#include "mpi_process.hpp"
#include "worker.hpp"
//Entry point for MPI master
void master_main(mpi_process& process){
    std::cout << process.world_rank << ": " << "Hello from master!" << std::endl;

    // distribute();

    // worker w(int world_rank, int world_size)
    // w.run

    // wait_for_result();
    // return;
}

void slave_main(mpi_process& process){
    std::cout << process.world_rank << ": " << "Hello from slave..." << std::endl;

    // receive_initial()

    // worker w(int world_rank, int world_size)
    // w.run

    // return
}



// MPI entry:
int main(){

    mpi_process process;

    if (process.world_rank==0) {

        master_main(process);
     
    }
    else{

        slave_main(process);

    }

    return 0;
}
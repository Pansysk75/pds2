#include <iostream>
#include <string>
#include <vector>

#include "detail/misc.hpp"
#include "detail/mpi_process.hpp"
#include "detail/communication.hpp"
#include "detail/knnDist.hpp"
#include "detail/worker.hpp"


//Entry point for MPI master
void master_main(mpi_process& process){

    com_port com(process.world_rank, process.world_size);

    int part_size = 20; 

    // Send initial data to all workers
    initial_work_data init_data{part_size,2,3};
    for(int i=1; i<process.world_size; i++){
        com.send(init_data, i);
    }
    // Initialize local worker
    worker w(process.world_rank, process.world_size, init_data);

    w.work();

    // Gather result
    std::vector<ResultPacket> diffProcRes;
    for(int i=1; i<process.world_size; i++){
        ResultPacket result;
        com.receive(result, i);
        diffProcRes.push_back(result);
    }

    ResultPacket final_result = ResultPacket::combineCompleteQueries(diffProcRes);

}

void slave_main(mpi_process& process){

    worker w(process.world_rank, process.world_size);
    w.work();

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
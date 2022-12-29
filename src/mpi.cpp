#include <iostream>
#include <string>
#include <vector>

#include "detail/mpi_process.hpp"
#include "detail/communication.hpp"
#include "detail/knnDist.hpp"
#include "detail/worker.hpp"
#include "detail/utilities.hpp"


//Entry point for MPI master
void master_main(mpi_process& process){

    com_port com(process.world_rank, process.world_size);

    int part_size = 20000; 

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
    diffProcRes.push_back(w.results);
    for(int i=1; i<process.world_size; i++){
        ResultPacket result(init_data.k);
        com.receive(result, i);
        diffProcRes.push_back(result);
    }

    std::cout << "RESULTS\n";
    for(auto& elem : diffProcRes){
        std::cout << elem.x_start_index << " " << elem.x_end_index << " | " << elem.y_start_index << " " << elem.y_end_index << " | " << elem.m_packet << " " << elem.n_packet << " | " << elem.k << std::endl;
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
        utilities::timer my_timer;
        my_timer.start();
        master_main(process);
        my_timer.stop();
        std::cout << "Total time: " << my_timer.get()/1000000.0 << " ms" << std::endl;
     
    }
    else{

        slave_main(process);

    }

    return 0;
}
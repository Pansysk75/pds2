#include <iostream>
#include <string>
#include <vector>

#include "detail/knn_utils.hpp"
#include "detail/mpi_process.hpp"

void test_com(mpi_process &proc)
{
    int id = proc.world_rank;
    int world_size = proc.world_size;

    com_port com(id, world_size);

    std::vector<size_t> data(10, id);
    size_t data2 = id;

    std::vector<size_t> recv_data(10);
    size_t recv_data2;

    // send stuff in a circle
    size_t next_rank = (id + 1) % world_size;
    size_t prev_rank = (id + world_size - 1) % world_size;

    std::cout << "\n\nStarting com test" << std::endl;

    std::cout << id << ": Sending to " << next_rank << ", receiving from " << prev_rank << std::endl;

    for (int _ = 0; _ < world_size + 2; _++)
    { // Repeat many times to expose potential issues
        // Data should do a full circle + 1
        com_request recv_req = com.receive_begin(prev_rank, recv_data, recv_data2);
        com_request send_req = com.send_begin(next_rank, data, data2);

        com.wait(send_req, recv_req);
        std::swap(data, recv_data);
        std::swap(data2, recv_data2);
    }

    bool success = true;
    for (auto &elem : recv_data){
      if(elem != next_rank) success = false;
    }
    if(recv_data2 != next_rank) success = false;
      
    std::cout << id << ": Communication test finished: " << (success?"success":"failure") << std::endl;
}

int main(int argc, char **argv) 
{

    mpi_process proc(&argc, &argv);

    if(proc.world_size == 1){
      std::cout << "Not executing test_com, as world size == 1 (launch more than one process with mpirun)" << std::endl;
    }else{
      test_com(proc);
    }

    return 0;
}

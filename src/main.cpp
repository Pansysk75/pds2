#include <iostream>
#include <string>
#include <vector>

#include "misc.hpp"
#include <mpi/mpi.h>
#include "mpi_process.hpp"
#include "communication.hpp"
// #include "worker.hpp"

//Entry point for MPI master
void master_main(mpi_process& process){
    std::cout << process.world_rank << ": " << "I'm master!" << std::endl;

    com_port port;


    //_____test blocking send_____
    int k = 42;
    std::cout << process.world_rank << ": " << "I'm sending " << k << std::endl;
    port.send(k, 2);


    // _____test non-blocking send_____
    std::vector<int> vec{0,1,2,3,4,5,6,7,8,9};

    // Print what we will send
    std::cout << process.world_rank << ": " << "Sending ";
    for(auto& elem : vec) std::cout << elem << " "; 
    std::cout << std::endl;

    // Begin sending
    std::vector<com_request> requests;
    for(auto& elem : vec){
        requests.push_back( port.send_begin(elem, 2) );
    }

    // do other stuff ...
    
    // Wait for everything to arrive before exiting
    for(auto& request : requests) port.wait(request);

}

void slave_main(mpi_process& process){
    std::cout << process.world_rank << ": " << "I'm slave" << std::endl;

    if(process.world_rank != 2) return;

    com_port port;
    
    // ____receive blocking____
    int k;
    port.receive(k, 0);
    std::cout << process.world_rank << ": " << "I received " << k << std::endl;

    // ____receive non-blocking____
    
    std::vector<int> vec(10);
    std::vector<com_request> requests(vec.size());
    for(int i=0; i<vec.size(); i++){
        requests[i] = port.receive_begin(vec[i], 0);
    }

    // Do other stuff...
    std::cout << process.world_rank << ": " << "I didn't block!"  << std::endl;

    // Wait for result to be ready
    for(auto& request : requests) port.wait(request);

    // Print what we received
    std::cout << process.world_rank << ": " << "I received ";
    for(auto& elem : vec) std::cout << elem << " "; 
    std::cout << std::endl;


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
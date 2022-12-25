#include <iostream>
#include <string>
#include <vector>

#include <mpi/mpi.h>

#include "misc.hpp"

#include "mpi_process.hpp"
// #include "communication.hpp"
#include "points_chunk.hpp"

//Entry point for MPI master
void master_main(mpi_process& process){
    std::cout << process.world_rank << ": " << "I'm master!" << std::endl;

    com_port port;


    //_____test blocking send_____
    points_chunk p{std::vector<double>{1,2,3,4,5,6,7,8,9}, 9, 2, 3};
    port.send(p, 2);


    // _____test non-blocking send_____

    // Begin sending
    com_request request;
    request = port.send_begin(p, 2);
    // do other stuff ...
    
    // Wait for everything to arrive before exiting
    port.wait(request);

}

void slave_main(mpi_process& process){
    std::cout << process.world_rank << ": " << "I'm slave" << std::endl;

    if(process.world_rank != 2) return;

    com_port port;
    
    // ____receive blocking____
    std::cout << process.world_rank << ": " << "Start receiving blocking"  << std::endl;
    points_chunk p{9, 0, 0};
    port.receive(p, 0);
    std::cout << process.world_rank << ": " << "I received: \n\t" << p.to_string() << std::endl;;


   
    // ____receive non-blocking____
    std::cout << process.world_rank << ": " << "Start receiving non-blocking!" << std::endl;
    points_chunk p2{9, 0, 0};
    com_request request;
    request = port.receive_begin(p2, 0);
    

    // Do other stuff...
    std::cout << process.world_rank << ": " << "I didn't block!"  << std::endl;

    // Wait for result to be ready
    port.wait(request);

    // Print what we received
    std::cout << process.world_rank << ": " << "I received: \n\t" << p2.to_string() << std::endl;;   
 


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
#include <iostream>
#include <mpi/mpi.h>

struct mpi_process{
    // Initializes MPI process instance and holds relevant data
    int world_size;
    int world_rank;

    mpi_process(){
        MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    }

    ~mpi_process(){
        std::cout << world_rank << ": exited" << std::endl;
        MPI_Finalize();
    }

    // Overload the "=" operator and the copy-constructor to prevent accidental 
    // copying. This object should be only created once and then just referenced.
    mpi_process& operator= (const mpi_process&) = delete;
    mpi_process (const mpi_process&) = delete;
};

#pragma once
#include "communication.hpp"
#include "points_chunk.hpp"

#define MASTER_RANK 0


struct initial_work_data{
    // Must be passed to every worker proccess, stores
    // data that is not known at compile time
    int n; // number of elements in each packet
    int d; // dimensionality of point-space
    int k; // number of nearest neighbours that should be found
};

// Could be made to take advantage of existing "com_port::send(int&, int)"
template<>
void com_port::send(initial_work_data &d, int receiver_rank){
    MPI_Send(&d.n, 1, MPI_INT, receiver_rank, 0, MPI_COMM_WORLD);
    MPI_Send(&d.d, 1, MPI_INT, receiver_rank, 1, MPI_COMM_WORLD);
    MPI_Send(&d.k, 1, MPI_INT, receiver_rank, 2, MPI_COMM_WORLD);
}

template<>
com_request com_port::send_begin(initial_work_data &d, int receiver_rank){
    com_request requests(3);
    MPI_Isend(&d.n, 1, MPI_INT, receiver_rank, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(&d.d, 1, MPI_INT, receiver_rank, 1, MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(&d.k, 1, MPI_INT, receiver_rank, 2, MPI_COMM_WORLD, &requests[2]);
    return requests;
}

template<>
void com_port::receive(initial_work_data &d, int sender_rank){
    MPI_Recv(&d.n, 1, MPI_INT, sender_rank, 0, MPI_COMM_WORLD, nullptr);
    MPI_Recv(&d.d, 1, MPI_INT, sender_rank, 1, MPI_COMM_WORLD, nullptr);
    MPI_Recv(&d.k, 1, MPI_INT, sender_rank, 2, MPI_COMM_WORLD, nullptr);
}

template<>
com_request com_port::receive_begin(initial_work_data &d, int sender_rank){
    com_request requests(3);
    MPI_Irecv(&d.n, 1, MPI_INT, sender_rank, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(&d.d, 1, MPI_INT, sender_rank, 1, MPI_COMM_WORLD, &requests[1]);
    MPI_Irecv(&d.k, 1, MPI_INT, sender_rank, 2, MPI_COMM_WORLD, &requests[2]);
    return com_request{requests};
}



class worker{

    public:
    // Communication port which facilitates all communication
    com_port com;

    // Initial data which will be received from a "master" process
    initial_work_data init_data;

    // Points this worker is responsible for (constant for the lifetime of this worker)
    points_chunk query_set;

    // Points which are processed in one iteration (different for every iteration)
    points_chunk corpus_set;

    // A set that is allocated for receiving a points_chunk.
    points_chunk receiving_set;

    worker(int rank, int world_size, bool is_local = false)
        :com(rank, world_size){
        
        // Initialize worker and receive inital data / corpus set
        // The master mpi process is responsible for transfering initial 
        // data to all workers. Then, the workers will exchange data in a 
        // cyclical pattern, until all workers have processed all data.

        if(!is_local){
            com.receive(init_data, MASTER_RANK);
            
            query_set = points_chunk(init_data.n, 0, 0);
            corpus_set = points_chunk(init_data.n, 0, 0);
            receiving_set = points_chunk(init_data.n, 0, 0);


            com.receive(query_set, MASTER_RANK);
            com.receive(corpus_set, MASTER_RANK);

            deb("Received initial data");
        }
    }

    void deb(std::string str){
        std::cout << com.rank() << ": " << str << std::endl;
    }



    void work(){

        for(int i=0; i<com.world_size(); i++){

            deb("Started iteration " + std::to_string(i));

            int next_rank = (com.rank() + 1) % com.world_size();
            int prev_rank = (com.rank() + com.world_size() - 1) % com.world_size();


            // Start sending the part we just proccessed
            // Start receiving the part we will proccess later
            com_request send_req = com.send_begin(corpus_set, next_rank);
            com_request recv_req = com.receive_begin(receiving_set, prev_rank);


            // Work on working set
            // kNN(query_set, corpus_set)
            // deb(corpus_set.to_string());

            // Wait for open communications to finish before
            // starting next iteration.
            com.wait(send_req);
            com.wait(recv_req);

            // Update query_set with received set (using std::swap is the
            // equivelant of swapping the pointers of two C arrays)
            std::swap(corpus_set, receiving_set);
            
        }

        // Work finished, send results to master process
        // com.send(knn_result, MASTER_RANK)

    }

};



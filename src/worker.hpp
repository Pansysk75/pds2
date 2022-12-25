#include "communication.hpp"

#define MASTER_RANK 0

class point_set{}; // Placeholder

struct initial_worker_data{
    // Must be passed to every worker proccess, stores
    // data that is not known at compile time

};

class worker{

    unsigned int rank;

    com_port com;

    point_set query_set;
    point_set corpus_set;

    point_set receiving_set;

    worker(unsigned int rank){
        // Initialize slave and receive corpus set
        // For now, the master mpi process is responsible
        // for transfering initial data to all slaves. Then, the slaves
        // will exchange that data in a cyclical pattern, until all
        // slaves have processed all data.
        this->rank = rank;

        receive_set(query_set, MASTER_RANK);
        receive_set(corpus_set, MASTER_RANK);
        
    }


    void work_loop(){

        for(i=0; i<world_size; i++){


            // Start sending the part we just proccessed
            // Start receiving the part we will proccess later
            cyclic_send_begin(corpus_set);
            cyclic_receive_begin(receiving_set);


            // Work on working set
            kNN(query_points, corpus_points); //TODO: Think about how kNN points will be updated every iteration


            // Wait for open communications to finish before
            // starting next iteration.
            cyclic_send_wait();
            cyclic_receive_wait();

            // Update query_set with received set
            corpus_set = std::move(receiving_set);
            
        }

    }

};



#pragma once
#include "communication.hpp"
#include "knnDist.hpp"

#define MASTER_RANK 0

std::vector<double> import_data(int idx_start, int idx_end){
    // Imitates importing data
    int size = idx_end - idx_start;
    std::vector<double> vec(size);
    std::iota(vec.begin(), vec.end(), idx_start);
    return vec;
}


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
    QueryPacket query;

    // Points which are processed in one iteration (different for every iteration)
    CorpusPacket corpus;

    // A set that is allocated for receiving a points_chunk.
    CorpusPacket receiving_corpus;

    // KNN result for local points, will be combined in master process
    ResultPacket results;

    worker(int rank, int world_size)
        :com(rank, world_size){       
        // Initialize worker and receive inital data / corpus set
        // The master mpi process is responsible for transfering initial 
        // data to all workers. Then, the workers will exchange data in a 
        // cyclical pattern, until all workers have processed all data.      
            com.receive(init_data, MASTER_RANK);
            init();
    }

    worker(int rank, int world_size, initial_work_data init_data)
        :com(rank, world_size), init_data(init_data){
        init();
    }
    
    void init(){
            int size = init_data.n;
            
            int idx_start = com.rank()*size;
            int idx_end = (com.rank()+1)*size;

            query = QueryPacket(size, init_data.d, idx_start, idx_end);
            query.X = import_data(idx_start, idx_end);

            corpus = CorpusPacket(size, init_data.d, idx_start, idx_end);
            corpus.Y = import_data(idx_start, idx_end);

            receiving_corpus.Y.resize(size);

            results = ResultPacket(size, size, init_data.k, idx_start, idx_end, idx_start, idx_end);

            deb("Initialization complete!");
    }

    void deb(std::string str){
        std::cout << com.rank() << ": " << str << std::endl;
    }



    void work(){

        deb("Starting work with sizes " + std::to_string(corpus.Y.size()));

        for(int i=0; i<com.world_size(); i++){

            deb("Started iteration " + std::to_string(i));

            int next_rank = (com.rank() + 1) % com.world_size();
            int prev_rank = (com.rank() + com.world_size() - 1) % com.world_size();


            // Start sending the part we just proccessed
            // Start receiving the part we will proccess later
            com_request send_req = com.send_begin(corpus, next_rank);
            com_request recv_req = com.receive_begin(receiving_corpus, prev_rank);


            // Work on working set
            deb("Started calculating batch " + std::to_string(i));
            ResultPacket batch_result(query, corpus, init_data.k);
            deb("Finished calculating batch " + std::to_string(i));
            // Combine this result with previous results
            results = ResultPacket::combineKnnResultsSameX(results, batch_result);

            // Wait for open communications to finish
            com.wait(send_req);
            com.wait(recv_req);

            // Update query_set with received set (using std::swap is the
            // equivelant of swapping the pointers of two C arrays)
            std::swap(corpus, receiving_corpus);
            
        }

        // Work finished, send results to master process
        if (com.rank() != MASTER_RANK) com.send(results, MASTER_RANK);

    }

};



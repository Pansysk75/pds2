#pragma once
#include <mpi/mpi.h>
#include <vector>


using com_request = std::vector<MPI_Request>;

class com_port{
    // Interface to send/receive data
    // NOTE: Currently does not work for sending/receiving multiple
    // objects simultaniously (one send and one receive at the same
    // time works).

    int _rank; // Rank(==id) of proccess where this com_port is created.
    int _world_size; // Number of processes we can communicate with.

public:
    // Constructor
    com_port(int rank, int world_size){ 
        _rank = rank; 
        _world_size = world_size;
    }

    int rank() const { return _rank; }
    int world_size() const {return _world_size;}

    // Expose sending/receiving interface:

    // Blocking receive
    template <typename T>
    void receive(T& obj, int sender_rank);

    // Non-blocking receive
    template <typename T>
    com_request receive_begin(T& obj, int sender_rank);

    // Blocking send
    template <typename T>
    void send(T& obj, int receiver_rank);

    // Non-blocking send
    template <typename T>
    com_request send_begin(T& obj, int receiver_rank);

    // Waits for a non-blocking communication to complete
    void wait(com_request request){
        for(auto& request_elem : request){
            MPI_Wait(&request_elem, nullptr);
        }
    };

};



// Implementation for int:

template<>
void com_port::send(int& k, int receiver_rank){
    MPI_Send(&k, 1, MPI_INT, receiver_rank, 0, MPI_COMM_WORLD);
}

template<>
com_request com_port::send_begin(int& k, int receiver_rank){
    MPI_Request request;
    MPI_Isend(&k, 1, MPI_INT, receiver_rank, 0, MPI_COMM_WORLD, &request);
    return com_request{request};
}

template<>
void com_port::receive(int& k, int sender_rank){
    MPI_Recv(&k, 1, MPI_INT, sender_rank, 0, MPI_COMM_WORLD, nullptr);
}

template<>
com_request com_port::receive_begin(int& k, int sender_rank){
    MPI_Request request;
    MPI_Irecv(&k, 1, MPI_INT, sender_rank, 0, MPI_COMM_WORLD, &request);
    return com_request{request};
}



// Implementation for std::vector<double>

template<>
void com_port::send(std::vector<double>& vec, int receiver_rank){
    MPI_Send(vec.data(), vec.size(), MPI_DOUBLE, receiver_rank, 0, MPI_COMM_WORLD);
}

template<>
com_request com_port::send_begin(std::vector<double>& vec, int receiver_rank){
    com_request requests(1);
    MPI_Isend(vec.data(), vec.size(), MPI_DOUBLE, receiver_rank, 0, MPI_COMM_WORLD, &requests[0]);
    return requests;
}

template<>
void com_port::receive(std::vector<double>& vec, int sender_rank){
    // Assume that c.data memory has already been initialized
    MPI_Recv(vec.data(), vec.size(), MPI_DOUBLE, sender_rank, 0, MPI_COMM_WORLD, nullptr);
}

template<>
com_request com_port::receive_begin(std::vector<double>& vec, int sender_rank){
    // Assume that c.data memory has already been initialized
    com_request requests(1);
    MPI_Irecv(vec.data(), vec.size(), MPI_DOUBLE, sender_rank, 0, MPI_COMM_WORLD, &requests[0]);
    return com_request{requests};
}

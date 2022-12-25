#include <mpi/mpi.h>
#include <vector>


using com_request = std::vector<MPI_Request>;

class com_port{
    // Interface to send/receive data

    // Haven't yet decided on how to implement this
    // We can probably get away with only implementing these
    // to work with one type (the one that describes a collection of k-dimensional points)

    public:

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


// Implementation for int type:

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


#include <vector>
#include "communication.hpp"

struct points_chunk{
    // Represents a chunk which is only part of the whole dataset.
    // idx_start and idx_end represent which part this is.
    public:
    std::vector<double> data;
    int size;
    int idx_start;
    int idx_end;

    points_chunk(int size, int idx_start, int idx_end)
    :size(size), idx_start(idx_start), idx_end(idx_end)
    {
        data.resize(size);
    }
    points_chunk(std::vector<double>&& vec,int size, int idx_start, int idx_end)
        :size(size), idx_start(idx_start), idx_end(idx_end)
    {
        data = std::move(vec);
    }

    std::string to_string(){
        std::string str;
        for(auto& elem : data){
            str += std::to_string(elem) + " ";
        }
       str += " | size=" + std::to_string(size) + " | idx_start=" + std::to_string(idx_start) + " | idx_end=" + std::to_string(idx_end);
       return str;
    }
};

template<>
void com_port::send(points_chunk& c, int receiver_rank){
    MPI_Send(c.data.data(), c.size, MPI_DOUBLE, receiver_rank, 0, MPI_COMM_WORLD);
    MPI_Send(&c.size, 1, MPI_INT, receiver_rank, 1, MPI_COMM_WORLD);
    MPI_Send(&c.idx_start, 1, MPI_INT, receiver_rank, 2, MPI_COMM_WORLD);
    MPI_Send(&c.idx_end, 1, MPI_INT, receiver_rank, 3, MPI_COMM_WORLD);
}

template<>
com_request com_port::send_begin(points_chunk& c, int receiver_rank){
    com_request requests(4);
    MPI_Isend(c.data.data(), c.size, MPI_DOUBLE, receiver_rank, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(&c.size, 1, MPI_INT, receiver_rank, 1, MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(&c.idx_start, 1, MPI_INT, receiver_rank, 2, MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(&c.idx_end, 1, MPI_INT, receiver_rank, 3, MPI_COMM_WORLD, &requests[3]);
    return requests;
}

template<>
void com_port::receive(points_chunk& c, int sender_rank){
    // lets assume that c.data memory has been initialized
    MPI_Recv(c.data.data(), c.size, MPI_DOUBLE, sender_rank, 0, MPI_COMM_WORLD, nullptr);
    MPI_Recv(&c.size, 1, MPI_INT, sender_rank, 1, MPI_COMM_WORLD, nullptr);
    MPI_Recv(&c.idx_start, 1, MPI_INT, sender_rank, 2, MPI_COMM_WORLD, nullptr);
    MPI_Recv(&c.idx_end, 1, MPI_INT, sender_rank, 3, MPI_COMM_WORLD, nullptr);
}

template<>
com_request com_port::receive_begin(points_chunk& c, int sender_rank){
    com_request requests(4);
    MPI_Irecv(c.data.data(), c.size, MPI_DOUBLE, sender_rank, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(&c.size, 1, MPI_INT, sender_rank, 1, MPI_COMM_WORLD, &requests[1]);
    MPI_Irecv(&c.idx_start, 1, MPI_INT, sender_rank, 2, MPI_COMM_WORLD, &requests[2]);
    MPI_Irecv(&c.idx_end, 1, MPI_INT, sender_rank, 3, MPI_COMM_WORLD, &requests[3]);
    return com_request{requests};
}

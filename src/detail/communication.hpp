#pragma once

#include <mpi/mpi.h>
#include <vector>

class com_request : public std::vector<MPI_Request>
{
    // A communicator request, will hold many MPI_Requests when
    // sending composite types
    // It's probably(?) ok to pass around MPI_Requests like that
public:
    com_request &operator<<(const com_request &rhs)
    {
        this->reserve(this->size() + rhs.size());
        this->insert(this->end(), rhs.begin(), rhs.end());
        return *this;
    }
    com_request(MPI_Request r)
    {
        this->push_back(r);
    }
};

class com_port
{
    // Interface to send/receive data
    // Contains multi-argument variadic templates with
    // specializations for intrinsinc types that we use (size_type and vector<double>).
    // The goal is (1) make it easier to send composite types (structs),
    // using the syntax "send(destination, a, b, c, ...)", where a,b,c,... can be
    // any combination of supported types and (2) make it easy to overload
    // "send(int destination, T)" for custom types.
    // ((It's overkill for this assignment but I wanted to play around with templates))

    // Internal value, needed to send multiple elements of the same type
    // without mixing them up on the receiving end
    unsigned int _tag;

    // Rank(==id) of proccess where this com_port is created.
    unsigned int _rank;

    // Number of processes we can communicate with.
    unsigned int _world_size;

public:
    // Constructor
    com_port(unsigned int rank, unsigned int world_size)
    {
        _tag = 0;
        _rank = rank;
        _world_size = world_size;
    }

    int rank() const { return _rank; }
    int world_size() const { return _world_size; }

    // Blocking receive
    template <typename T>
    void receive(int source_id, T &t);

    template <typename... P>
        requires(sizeof...(P) > 1)
    void receive(int source_id, P &...p)
    {
        (receive(source_id, p), ...);
        std::cout << "PROC\t" << _rank << "\tRECV " << _tag << std::endl;
        _tag = 0; // reset tag
    }

    // Non-blocking receive
    template <typename T>
    com_request receive_begin(int source_id, T &);

    template <typename... P>
        requires(sizeof...(P) > 1)
    com_request receive_begin(int source_id, P &...p)
    {
        com_request request = (receive_begin(source_id, p) << ...);
        std::cout << "PROC\t" << _rank << "\tISEND\t" << _tag << std::endl;
        _tag = 0; // reset tag
        return request;
    }

    // Blocking send
    template <typename T>
    void send(int source_id, T &);

    template <typename... P>
        requires(sizeof...(P) > 1)
    void send(int destination_id, P &...p)
    {
        (send(destination_id, p), ...);
        std::cout << "PROC\t" << _rank << "\tSEND\t" << _tag << std::endl;
        _tag = 0; // reset tag
    }

    // Non-blocking send
    template <typename T>
    com_request send_begin(int source_id, T &);

    template <typename... P>
        requires(sizeof...(P) > 1)
    com_request send_begin(int destination_id, P &...p)
    {
        com_request request = (send_begin(destination_id, p) << ...);
        std::cout << "PROC\t" << _rank << "\tIRECV\t" << _tag << std::endl;
        _tag = 0;
        return request;
    }

    // com_request send_begin(int destination_id, size_t &k);
    // com_request send_begin(int destination_id, std::vector<double> &k);

    // Waits for a non-blocking communication to complete
    void wait(com_request request)
    {
        for (auto &request_elem : request)
        {
            MPI_Wait(&request_elem, nullptr);
        }
    };
};

// Implement send/receive for the intrinsic types that we will use
// If we had compile-time access to MPI_INT, MPI_DOUBlE etc, this
// could have been much more elegant

////////////////
/// RECEIVE ///
//////////////

template <>
inline void com_port::receive(int source_id, int &k)
{
    MPI_Recv(&k, 1, MPI_INT, source_id, _tag++, MPI_COMM_WORLD, nullptr);
}

template <>
inline void com_port::receive(int source_id, size_t &k)
{
    MPI_Recv(&k, 1, MPI_UNSIGNED_LONG, source_id, _tag++, MPI_COMM_WORLD, nullptr);
}

template <>
inline void com_port::receive(int source_id, std::vector<double> &v)
{
    MPI_Recv(v.data(), v.size(), MPI_DOUBLE, source_id, _tag++, MPI_COMM_WORLD, nullptr);
}

template <>
inline void com_port::receive(int source_id, std::vector<size_t> &v)
{
    MPI_Recv(v.data(), v.size(), MPI_UNSIGNED_LONG, source_id, _tag++, MPI_COMM_WORLD, nullptr);
}

/////////////////////////////
/// NON-BLOCKING RECEIVE ///
///////////////////////////

template <>
inline com_request com_port::receive_begin(int source_id, int &k)
{
    MPI_Request request;
    MPI_Irecv(&k, 1, MPI_INT, source_id, _tag++, MPI_COMM_WORLD, &request);
    return com_request{request};
}

template <>
inline com_request com_port::receive_begin(int source_id, size_t &k)
{
    MPI_Request request;
    MPI_Irecv(&k, 1, MPI_UNSIGNED_LONG, source_id, _tag++, MPI_COMM_WORLD, &request);
    return com_request{request};
}

template <>
inline com_request com_port::receive_begin(int source_id, std::vector<double> &v)
{
    MPI_Request request;
    MPI_Irecv(v.data(), v.size(), MPI_DOUBLE, source_id, _tag++, MPI_COMM_WORLD, &request);
    return com_request{request};
}

template <>
inline com_request com_port::receive_begin(int source_id, std::vector<size_t> &v)
{
    MPI_Request request;
    MPI_Irecv(v.data(), v.size(), MPI_UNSIGNED_LONG, source_id, _tag++, MPI_COMM_WORLD, &request);
    return com_request{request};
}

/////////////
/// SEND ///
///////////

template <>
inline void com_port::send(int destination_id, int &k)
{
    MPI_Send(&k, 1, MPI_INT, destination_id, _tag++, MPI_COMM_WORLD);
}

template <>
inline void com_port::send(int destination_id, size_t &k)
{
    MPI_Send(&k, 1, MPI_UNSIGNED_LONG, destination_id, _tag++, MPI_COMM_WORLD);
}

template <>
inline void com_port::send(int destination_id, std::vector<double> &v)
{
    MPI_Send(v.data(), v.size(), MPI_DOUBLE, destination_id, _tag++, MPI_COMM_WORLD);
}

template <>
inline void com_port::send(int destination_id, std::vector<size_t> &v)
{
    MPI_Send(v.data(), v.size(), MPI_UNSIGNED_LONG, destination_id, _tag++, MPI_COMM_WORLD);
}

//////////////////////////
/// NON_BLOCKING SEND ///
////////////////////////

template <>
inline com_request com_port::send_begin(int destination_id, int &k)
{
    MPI_Request request;
    MPI_Isend(&k, 1, MPI_INT, destination_id, _tag++, MPI_COMM_WORLD, &request);
    return com_request{request};
}

template <>
inline com_request com_port::send_begin(int destination_id, size_t &k)
{
    MPI_Request request;
    MPI_Isend(&k, 1, MPI_UNSIGNED_LONG, destination_id, _tag++, MPI_COMM_WORLD, &request);
    return com_request{request};
}

template <>
inline com_request com_port::send_begin(int destination_id, std::vector<double> &v)
{
    MPI_Request request;
    MPI_Isend(v.data(), v.size(), MPI_DOUBLE, destination_id, _tag++, MPI_COMM_WORLD, &request);
    return com_request{request};
}

template <>
inline com_request com_port::send_begin(int destination_id, std::vector<size_t> &v)
{
    MPI_Request request;
    MPI_Isend(v.data(), v.size(), MPI_UNSIGNED_LONG, destination_id, _tag++, MPI_COMM_WORLD, &request);
    return com_request{request};
}
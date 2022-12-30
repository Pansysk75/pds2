#pragma once

#include "communication.hpp"
#include <vector>

struct CorpusPacket
{
    size_t n_packet;
    size_t d;

    size_t y_start_index;
    size_t y_end_index;

    std::vector<double> Y;

    // empty constructor
    CorpusPacket() {}

    CorpusPacket(size_t n_packet, size_t d, size_t y_start_index,
                 size_t y_end_index)
        : n_packet(n_packet), d(d), y_start_index(y_start_index),
          y_end_index(y_end_index)
    {
        Y = std::vector<double>(n_packet * d);
    }

    // moving Y into the packet
    CorpusPacket(size_t n_packet, size_t d, size_t y_start_index,
                 size_t y_end_index, std::vector<double> &&Y)
        : n_packet(n_packet), d(d), y_start_index(y_start_index),
          y_end_index(y_end_index), Y(std::move(Y)) {}
};

struct QueryPacket
{
    size_t m_packet;
    size_t d;

    size_t x_start_index;
    size_t x_end_index;

    std::vector<double> X;

    // empty constructor
    QueryPacket() {}

    QueryPacket(size_t m_packet, size_t d, size_t x_start_index,
                size_t x_end_index)
        : m_packet(m_packet), d(d), x_start_index(x_start_index),
          x_end_index(x_end_index)
    {
        X = std::vector<double>(m_packet * d);
    }

    // moving X into the packet
    QueryPacket(size_t m_packet, size_t d, size_t x_start_index,
                size_t x_end_index, std::vector<double> &&X)
        : m_packet(m_packet), d(d), x_start_index(x_start_index),
          x_end_index(x_end_index), X(std::move(X)) {}
};

struct ResultPacket
{

    size_t m_packet;
    size_t n_packet;
    size_t k;

    size_t x_start_index;
    size_t x_end_index;

    // if y_end_index < y_start_index, then it wraps around
    // for example if y_start_index = 0 and y_end_index = 1000, then it is the
    // first 1000 points if y_start_index = 200 and y_end_index = 100 and
    // n_packet = 500 then it is the points 200:600 and 0:100
    size_t y_start_index;
    size_t y_end_index;

    // in global index of y
    std::vector<size_t> nidx;
    std::vector<double> ndist;

    // empty constructor
    ResultPacket() {}

    ResultPacket(size_t vec_size)
    {
        nidx.resize(vec_size);
        ndist.resize(vec_size);
        m_packet = 0;
        n_packet = 0;
        k = 0;
        x_start_index = 0;
        x_end_index = 0;
        y_start_index = 0;
        y_end_index = 0;
    }

    // this is the constructor for the result packet, without it being solved.
    // It needs to be filled manually
    ResultPacket(size_t m_packet, size_t n_packet, size_t k_arg,
                 size_t x_start_index, size_t x_end_index,
                 size_t y_start_index, size_t y_end_index)
        : m_packet(m_packet), n_packet(n_packet), k(std::min(k_arg, n_packet)),
          x_start_index(x_start_index), x_end_index(x_end_index),
          y_start_index(y_start_index), y_end_index(y_end_index)
    {
        nidx = std::vector<size_t>(m_packet * this->k);
        ndist = std::vector<double>(m_packet * this->k);
    }

    // this is the solver, it takes a query and a corpus and returns a result
    ResultPacket(const QueryPacket &query, const CorpusPacket &corpus,
                 size_t k);
};

// CorpusPacket send/receive (This is ugly and wrong in so
// many ways, i know)

template <>
inline void com_port::send(CorpusPacket &c, int receiver_rank)
{
    MPI_Send(c.Y.data(), c.n_packet * c.d, MPI_DOUBLE, receiver_rank, 0,
             MPI_COMM_WORLD);
    MPI_Send(&c.n_packet, 1, MPI_INT, receiver_rank, 1, MPI_COMM_WORLD);
    MPI_Send(&c.d, 1, MPI_INT, receiver_rank, 2, MPI_COMM_WORLD);
    MPI_Send(&c.y_start_index, 1, MPI_INT, receiver_rank, 3, MPI_COMM_WORLD);
    MPI_Send(&c.y_end_index, 1, MPI_INT, receiver_rank, 4, MPI_COMM_WORLD);
}

template <>
inline com_request com_port::send_begin(CorpusPacket &c, int receiver_rank)
{
    com_request requests(5);
    MPI_Isend(c.Y.data(), c.n_packet * c.d, MPI_DOUBLE, receiver_rank, 0,
              MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(&c.n_packet, 1, MPI_INT, receiver_rank, 1, MPI_COMM_WORLD,
              &requests[1]);
    MPI_Isend(&c.d, 1, MPI_INT, receiver_rank, 2, MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(&c.y_start_index, 1, MPI_INT, receiver_rank, 3, MPI_COMM_WORLD,
              &requests[3]);
    MPI_Isend(&c.y_end_index, 1, MPI_INT, receiver_rank, 4, MPI_COMM_WORLD,
              &requests[4]);

    return requests;
}

template <>
inline void com_port::receive(CorpusPacket &c, int sender_rank)
{
    // Assume that c.data memory has already been initialized
    MPI_Recv(c.Y.data(), c.Y.size(), MPI_DOUBLE, sender_rank, 0, MPI_COMM_WORLD,
             nullptr);
    MPI_Recv(&c.n_packet, 1, MPI_INT, sender_rank, 1, MPI_COMM_WORLD, nullptr);
    MPI_Recv(&c.d, 1, MPI_INT, sender_rank, 2, MPI_COMM_WORLD, nullptr);
    MPI_Recv(&c.y_start_index, 1, MPI_INT, sender_rank, 3, MPI_COMM_WORLD,
             nullptr);
    MPI_Recv(&c.y_end_index, 1, MPI_INT, sender_rank, 4, MPI_COMM_WORLD,
             nullptr);
}

template <>
inline com_request com_port::receive_begin(CorpusPacket &c, int sender_rank)
{
    // Assume that c.data memory has already been initialized
    com_request requests(5);
    MPI_Irecv(c.Y.data(), c.Y.size(), MPI_DOUBLE, sender_rank, 0,
              MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(&c.n_packet, 1, MPI_INT, sender_rank, 1, MPI_COMM_WORLD,
              &requests[1]);
    MPI_Irecv(&c.d, 1, MPI_INT, sender_rank, 2, MPI_COMM_WORLD, &requests[2]);
    MPI_Irecv(&c.y_start_index, 1, MPI_INT, sender_rank, 3, MPI_COMM_WORLD,
              &requests[3]);
    MPI_Irecv(&c.y_end_index, 1, MPI_INT, sender_rank, 4, MPI_COMM_WORLD,
              &requests[4]);

    return com_request{requests};
}

// QueryPacket send/receive (This is ugly and wrong in so
// many ways, i know)

template <>
inline void com_port::send(QueryPacket &c, int receiver_rank)
{
    MPI_Send(c.X.data(), c.m_packet * c.d, MPI_DOUBLE, receiver_rank, 0,
             MPI_COMM_WORLD);
    MPI_Send(&c.m_packet, 1, MPI_INT, receiver_rank, 1, MPI_COMM_WORLD);
    MPI_Send(&c.d, 1, MPI_INT, receiver_rank, 2, MPI_COMM_WORLD);
    MPI_Send(&c.x_start_index, 1, MPI_INT, receiver_rank, 3, MPI_COMM_WORLD);
    MPI_Send(&c.x_end_index, 1, MPI_INT, receiver_rank, 4, MPI_COMM_WORLD);
}

template <>
inline com_request com_port::send_begin(QueryPacket &c, int receiver_rank)
{
    com_request requests(5);
    MPI_Isend(c.X.data(), c.m_packet * c.d, MPI_DOUBLE, receiver_rank, 0,
              MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(&c.m_packet, 1, MPI_INT, receiver_rank, 1, MPI_COMM_WORLD,
              &requests[1]);
    MPI_Isend(&c.d, 1, MPI_INT, receiver_rank, 2, MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(&c.x_start_index, 1, MPI_INT, receiver_rank, 3, MPI_COMM_WORLD,
              &requests[3]);
    MPI_Isend(&c.x_end_index, 1, MPI_INT, receiver_rank, 4, MPI_COMM_WORLD,
              &requests[4]);

    return requests;
}

template <>
inline void com_port::receive(QueryPacket &c, int sender_rank)
{
    // Assume that c.data memory has already been initialized
    MPI_Recv(c.X.data(), c.X.size(), MPI_DOUBLE, sender_rank, 0, MPI_COMM_WORLD,
             nullptr);
    MPI_Recv(&c.m_packet, 1, MPI_INT, sender_rank, 1, MPI_COMM_WORLD, nullptr);
    MPI_Recv(&c.d, 1, MPI_INT, sender_rank, 2, MPI_COMM_WORLD, nullptr);
    MPI_Recv(&c.x_start_index, 1, MPI_INT, sender_rank, 3, MPI_COMM_WORLD,
             nullptr);
    MPI_Recv(&c.x_end_index, 1, MPI_INT, sender_rank, 4, MPI_COMM_WORLD,
             nullptr);
}

template <>
inline com_request com_port::receive_begin(QueryPacket &c, int sender_rank)
{
    // Assume that c.data memory has already been initialized
    com_request requests(5);
    MPI_Irecv(c.X.data(), c.X.size(), MPI_DOUBLE, sender_rank, 0,
              MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(&c.m_packet, 1, MPI_INT, sender_rank, 1, MPI_COMM_WORLD,
              &requests[1]);
    MPI_Irecv(&c.d, 1, MPI_INT, sender_rank, 2, MPI_COMM_WORLD, &requests[2]);
    MPI_Irecv(&c.x_start_index, 1, MPI_INT, sender_rank, 3, MPI_COMM_WORLD,
              &requests[3]);
    MPI_Irecv(&c.x_end_index, 1, MPI_INT, sender_rank, 4, MPI_COMM_WORLD,
              &requests[4]);

    return com_request{requests};
}

// ResultsPacket send/receive (This is ugly and wrong in so
// many ways, i know)

template <>
inline void com_port::send(ResultPacket &c, int receiver_rank)
{
    MPI_Send(c.nidx.data(), c.k, MPI_INT, receiver_rank, 0, MPI_COMM_WORLD);
    MPI_Send(c.ndist.data(), c.k, MPI_DOUBLE, receiver_rank, 1, MPI_COMM_WORLD);

    MPI_Send(&c.m_packet, 1, MPI_INT, receiver_rank, 2, MPI_COMM_WORLD);
    MPI_Send(&c.n_packet, 1, MPI_INT, receiver_rank, 3, MPI_COMM_WORLD);

    MPI_Send(&c.k, 1, MPI_INT, receiver_rank, 4, MPI_COMM_WORLD);
    MPI_Send(&c.x_start_index, 1, MPI_INT, receiver_rank, 5, MPI_COMM_WORLD);
    MPI_Send(&c.x_end_index, 1, MPI_INT, receiver_rank, 6, MPI_COMM_WORLD);
    MPI_Send(&c.y_start_index, 1, MPI_INT, receiver_rank, 7, MPI_COMM_WORLD);
    MPI_Send(&c.y_end_index, 1, MPI_INT, receiver_rank, 8, MPI_COMM_WORLD);
}

template <>
inline com_request com_port::send_begin(ResultPacket &c, int receiver_rank)
{
    com_request requests(9);
    MPI_Isend(c.nidx.data(), c.nidx.size(), MPI_INT, receiver_rank, 0,
              MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(c.ndist.data(), c.ndist.size(), MPI_DOUBLE, receiver_rank, 1,
              MPI_COMM_WORLD, &requests[1]);

    MPI_Isend(&c.m_packet, 1, MPI_INT, receiver_rank, 2, MPI_COMM_WORLD,
              &requests[2]);
    MPI_Isend(&c.n_packet, 1, MPI_INT, receiver_rank, 3, MPI_COMM_WORLD,
              &requests[3]);

    MPI_Isend(&c.k, 1, MPI_INT, receiver_rank, 4, MPI_COMM_WORLD, &requests[4]);
    MPI_Isend(&c.x_start_index, 1, MPI_INT, receiver_rank, 5, MPI_COMM_WORLD,
              &requests[5]);
    MPI_Isend(&c.x_end_index, 1, MPI_INT, receiver_rank, 6, MPI_COMM_WORLD,
              &requests[6]);
    MPI_Isend(&c.y_start_index, 1, MPI_INT, receiver_rank, 7, MPI_COMM_WORLD,
              &requests[7]);
    MPI_Isend(&c.y_end_index, 1, MPI_INT, receiver_rank, 8, MPI_COMM_WORLD,
              &requests[8]);
    return requests;
}

template <>
inline void com_port::receive(ResultPacket &c, int sender_rank)
{
    // Assume that c.data memory has already been initialized
    MPI_Recv(c.nidx.data(), c.nidx.size(), MPI_INT, sender_rank, 0,
             MPI_COMM_WORLD, nullptr);
    MPI_Recv(c.ndist.data(), c.ndist.size(), MPI_DOUBLE, sender_rank, 1,
             MPI_COMM_WORLD, nullptr);

    MPI_Recv(&c.m_packet, 1, MPI_INT, sender_rank, 2, MPI_COMM_WORLD, nullptr);
    MPI_Recv(&c.n_packet, 1, MPI_INT, sender_rank, 3, MPI_COMM_WORLD, nullptr);

    MPI_Recv(&c.k, 1, MPI_INT, sender_rank, 4, MPI_COMM_WORLD, nullptr);
    MPI_Recv(&c.x_start_index, 1, MPI_INT, sender_rank, 5, MPI_COMM_WORLD,
             nullptr);
    MPI_Recv(&c.x_end_index, 1, MPI_INT, sender_rank, 6, MPI_COMM_WORLD,
             nullptr);
    MPI_Recv(&c.y_start_index, 1, MPI_INT, sender_rank, 7, MPI_COMM_WORLD,
             nullptr);
    MPI_Recv(&c.y_end_index, 1, MPI_INT, sender_rank, 8, MPI_COMM_WORLD,
             nullptr);
}

template <>
inline com_request com_port::receive_begin(ResultPacket &c, int sender_rank)
{
    // Assume that c.data memory has already been initialized
    com_request requests(9);
    MPI_Irecv(c.nidx.data(), c.nidx.size(), MPI_INT, sender_rank, 0,
              MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(c.ndist.data(), c.ndist.size(), MPI_DOUBLE, sender_rank, 1,
              MPI_COMM_WORLD, &requests[1]);

    MPI_Irecv(&c.m_packet, 1, MPI_INT, sender_rank, 2, MPI_COMM_WORLD,
              &requests[2]);
    MPI_Irecv(&c.n_packet, 1, MPI_INT, sender_rank, 3, MPI_COMM_WORLD,
              &requests[3]);

    MPI_Irecv(&c.k, 1, MPI_INT, sender_rank, 4, MPI_COMM_WORLD, &requests[4]);
    MPI_Irecv(&c.x_start_index, 1, MPI_INT, sender_rank, 5, MPI_COMM_WORLD,
              &requests[5]);
    MPI_Irecv(&c.x_end_index, 1, MPI_INT, sender_rank, 6, MPI_COMM_WORLD,
              &requests[6]);
    MPI_Irecv(&c.y_start_index, 1, MPI_INT, sender_rank, 7, MPI_COMM_WORLD,
              &requests[7]);
    MPI_Irecv(&c.y_end_index, 1, MPI_INT, sender_rank, 8, MPI_COMM_WORLD,
              &requests[8]);

    return com_request{requests};
}
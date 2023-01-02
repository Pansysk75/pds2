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

// make printable

inline std::ostream& operator<<(std::ostream& os, const CorpusPacket& c)
{
    os <<  c.y_start_index<< "->" << c.y_end_index << " | n:" << c.n_packet << " d:" << c.d;
    return os;
}

// make sendable

template <>
inline com_request com_port::_impl_send_begin(int destination_id, CorpusPacket &c)
{
    return send_begin(destination_id, c.d, c.n_packet, c.y_start_index, c.y_end_index, c.Y);
}

// make receivable

template <>
inline com_request com_port::_impl_receive_begin(int source_id, CorpusPacket &c)
{
    return receive_begin(source_id, c.d, c.n_packet, c.y_start_index, c.y_end_index, c.Y);
}

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

// make printable

inline std::ostream& operator<<(std::ostream& os, const QueryPacket& c)
{
    os <<  c.x_start_index<< "->" << c.x_end_index << " | n:" << c.m_packet << " d:" << c.d;
    return os;
}

// make sendable

template <>
inline com_request com_port::_impl_send_begin(int destination_id, QueryPacket &c)
{
    return send_begin(destination_id, c.d, c.m_packet, c.x_start_index, c.x_end_index, c.X);
}

// make receivable

template <>
inline com_request com_port::_impl_receive_begin(int source_id, QueryPacket &c)
{
    return receive_begin(source_id, c.d, c.m_packet, c.x_start_index, c.x_end_index, c.X);
}

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

// make printable
inline std::ostream& operator<<(std::ostream& os, const ResultPacket& c)
{
    os << "X:" << c.x_start_index<< "->" << c.x_end_index << " | Y:" << c.y_start_index<< "->" << c.y_end_index;
    return os;
}

// make sendable

template <>
inline void com_port::_impl_send(int destination_id, ResultPacket &c)
{
    send(destination_id, c.k, c.m_packet, c.n_packet, c.ndist, c.nidx, c.x_end_index, c.x_start_index, c.y_end_index, c.y_start_index);
}

// make receivable

template <>
inline void com_port::_impl_receive(int destination_id, ResultPacket &c)
{
    receive(destination_id, c.k, c.m_packet, c.n_packet, c.ndist, c.nidx, c.x_end_index, c.x_start_index, c.y_end_index, c.y_start_index);
}

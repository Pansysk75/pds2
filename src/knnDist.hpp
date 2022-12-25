#pragma once

#include <iostream>

#include <vector>
#include <tuple>

#define idx(i, j, d) ((i)*(d) + (j))

struct CorpusPacket {
    const size_t n_packet;
    const size_t d;

    size_t y_start_index;
    size_t y_end_index;

    std::vector<double> Y;

    // moving Y into the packet
    CorpusPacket(
        size_t n_packet, size_t d,
        size_t y_start_index, size_t y_end_index,
        std::vector<double>&& Y
    );

    CorpusPacket(
        size_t n_packet, size_t d,
        size_t y_start_index, size_t y_end_index
    );
};

struct QueryPacket {
    const size_t m_packet;
    const size_t d;

    size_t x_start_index;
    size_t x_end_index;

    std::vector<double> X;

    // moving X into the packet
    QueryPacket(
        size_t m_packet, size_t d,
        size_t x_start_index, size_t x_end_index,
        std::vector<double>&& X
    );

    QueryPacket(
        size_t m_packet, size_t d,
        size_t x_start_index, size_t x_end_index
    );
};

struct ResultPacket {

    const size_t m_packet;
    const size_t n_packet;
    const size_t k;

    const size_t x_start_index;
    const size_t x_end_index;

    // if y_end_index < y_start_index, then it wraps around
    // for example if y_start_index = 0 and y_end_index = 1000, then it is the first 1000 points
    // if y_start_index = 200 and y_end_index = 100 and n_packet = 500 then it is the points 200:600 and 0:100
    const size_t y_start_index;
    const size_t y_end_index;

// in global index of y
    std::vector<size_t> nidx;
    std::vector<double> ndist;

    // this is the constructor for the result packet, without it being solved. It needs to be filled manually
    ResultPacket(
        size_t m_packet, size_t n_packet, size_t k,
        size_t x_start_index, size_t x_end_index,
        size_t y_start_index, size_t y_end_index
    );

    // this is the solver, it takes a query and a corpus and returns a result
    ResultPacket(
        const QueryPacket& query,
        const CorpusPacket& corpus,
        size_t k
    );

// they need to be distances of
// SAME query points
// DIFFERENT corpus points  
    static std::tuple<bool, size_t, size_t> combinableSameX(const ResultPacket& back, const ResultPacket& front);

// they need to be distances of
// DIFFERENT query points
// SAME corpus points
    static std::tuple<bool, size_t, size_t> combinableSameY(const ResultPacket& back, const ResultPacket& front);

    // they need to be combinableSameX have back.x_end_index == front.x_start_index
    // for example we combine the k nearest neighbors of x[0:100] in both results
    // but the first is the k nearest neighbors from y[0:100] and the second is the k nearest neighbors from y[100:200]
    static ResultPacket combineKnnResultsSameX(const ResultPacket& back, const ResultPacket& front);

/*
    // they need to be combinableSameY have back.y_end_index == front.y_start_index
    // for example we combine the k nearest neighbors from y[0:100] in both results
    // the first is the k nearest neighbors of x[0:100] and the second is the k nearest neighbors of x[100:200]
    static ResultPacket combineKnnResultsSameY(const ResultPacket& back, const ResultPacket& front);
*/

    // they all share the same Y (which is the whole Y) and collectivly cover the whole X
    static ResultPacket combineCompleteQueries(std::vector<ResultPacket&> results);
};
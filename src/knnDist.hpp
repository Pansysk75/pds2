#include <iostream>

#include <vector>

#include <algorithm>
#include <numeric> 

#include <cblas.h>

#define idx(i, j, d) ((i)*(d) + (j))

struct CorpusPacket {
    const size_t n_packet;
    const size_t d;

    size_t y_start_index;
    size_t y_end_index;

    std::vector<double> Y;

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

    QueryPacket(
        size_t m_packet, size_t d,
        size_t x_start_index, size_t x_end_index
    );
};

struct ResultPacket {
    const size_t m_packet;
    const size_t k;

    const size_t x_start_index;
    const size_t x_end_index;

    const size_t y_start_index;
    const size_t y_end_index;

    // in global index of y
    std::vector<size_t> nidx;
    std::vector<double> ndist;

    // this is the constructor for the result packet, without it being solved. It needs to be filled manually
    ResultPacket(
        size_t m_packet, size_t k,
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
    static bool xCombinable(const ResultPacket& p1, const ResultPacket& p2);

// they need to be distances of
// DIFFERENT query points
// SAME corpus points
    static bool yCombinable(const ResultPacket& p1, const ResultPacket& p2);

    // they need to be xCombinable have p1.x_end_index == p2.x_start_index
    // for example we combine the k nearest neighbors of x[0:100] in both results
    // but the first is the k nearest neighbors from y[0:100] and the second is the k nearest neighbors from y[100:200]
    static ResultPacket combineKnnResultsSameX(const ResultPacket& p1, const ResultPacket& p2);

    // they need to be yCombinable have p1.y_end_index == p2.y_start_index
    // for example we combine the k nearest neighbors from y[0:100] in both results
    // the first is the k nearest neighbors of x[0:100] and the second is the k nearest neighbors of x[100:200]
    static ResultPacket combineKnnResultsSameY(const ResultPacket& p1, const ResultPacket& p2);
};
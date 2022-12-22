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
    ) : 
        n_packet(n_packet), d(d),
        y_start_index(y_start_index), y_end_index(y_end_index)
    {
        Y = std::vector<double>(n_packet * d);
    }
};

struct QueryPacket {
    const size_t m_packet;
    const size_t d;

    size_t x_start_index;
    size_t x_end_index;

    std::vector<double> X;

    QueryPacket(
        size_t n_packet, size_t d,
        size_t x_start_index, size_t x_end_index
    ) : 
        m_packet(n_packet), d(d),
        x_start_index(x_start_index), x_end_index(x_end_index) 
    {
        X = std::vector<double>(m_packet * d);
    }
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

    ResultPacket(
        size_t m_packet, size_t k,
        size_t x_start_index, size_t x_end_index,
        size_t y_start_index, size_t y_end_index
    ) :
        m_packet(m_packet), k(k),
        x_start_index(x_start_index), x_end_index(x_end_index),
        y_start_index(y_start_index), y_end_index(y_end_index)
    {
        nidx = std::vector<size_t>(m_packet * k);
        ndist = std::vector<double>(m_packet * k);
    }

    // they need to be distances of the 
    // SAME query points
    // DIFFERENT corpus points  
    static bool xCombinable(ResultPacket const& p1, ResultPacket const& p2) {
        return (
            p1.k == p2.k &&
            p1.m_packet == p2.m_packet &&
            p1.x_start_index == p2.x_start_index &&
            p1.x_end_index == p2.x_end_index
        );
    }

    // they need to be distances of
    // DIFFERENT query points
    // SAME corpus points  
    static bool yCombinable(ResultPacket const& p1, ResultPacket const& p2) {
        return (
            p1.k == p2.k &&
            p1.y_start_index == p2.y_start_index && 
            p1.y_end_index == p2.y_end_index
        );
    }
};

ResultPacket knnPacket(CorpusPacket& corpus, QueryPacket& query, size_t k){
    ResultPacket result(
        query.m_packet, k,
        query.x_start_index, query.x_end_index,
        corpus.y_start_index, corpus.y_end_index
    );

    std::vector<double> X2(query.m_packet);
    for(size_t i = 0; i < query.m_packet; i++){
        X2[i] = cblas_ddot(
            query.d,
            &query.X[idx(i, 0, query.d)], 1,
            &query.X[idx(i, 0, query.d)], 1
        );
    }

    std::vector<double> Y2(corpus.n_packet);
    for(size_t i = 0; i < corpus.n_packet; i++){
        Y2[i] = cblas_ddot(
            corpus.d,
            &corpus.Y[idx(i, 0, corpus.d)], 1,
            &corpus.Y[idx(i, 0, corpus.d)], 1
        );
    }

    std::vector<double> D(query.m_packet * corpus.n_packet);

    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        query.m_packet, corpus.n_packet, corpus.d,
        -2.0,
        query.X.data(), query.d,
        corpus.Y.data(), corpus.d,
        0.0,
        D.data(), corpus.n_packet
    );

    for(size_t i = 0; i < query.m_packet; i++){
        // maybe cache X2_i
        // X2_i = X2[i];
        for(size_t j = 0; j < corpus.n_packet; j++){
            D[idx(i, j, corpus.n_packet)] += X2[i] + Y2[j];
        }
    }

/* ALTERNATIVELY 
    for(size_t i = 0; i < query.m_packet; i++){
        // maybe cache X2_i
        // X2_i = X2[i];
        for(size_t j = 0; j < corpus.n_packet; j++){
           D[idx(i, j, corpus.n_packet)] = X2[i] + Y2[j];
        }
    }

    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        query.m_packet, corpus.n_packet, corpus.d,
        -2.0,
        query.X.data(), query.d,
        corpus.Y.data(), corpus.d,
        1.0,
        D.data(), corpus.n_packet
    );
*/
    result.nidx.resize(query.m_packet * k);
    result.ndist.resize(query.m_packet * k);

    for(size_t i = 0; i < query.m_packet; i++) {
        std::vector<size_t> indices(corpus.n_packet);
        std::iota(indices.begin(), indices.end(), 0);

        size_t n_packet = corpus.n_packet;

        std::partial_sort_copy(
            indices.begin(), indices.end(),
            &result.nidx[idx(i, 0, k)], &result.nidx[idx(i, k, k)],
            [&D, i, n_packet](size_t a, size_t b) {
                return D[idx(i, a, n_packet)] < D[idx(i, b, n_packet)]; 
            }
        );

        // convert y index to global y index
        for(size_t j = 0; j < k; j++) {
            result.nidx[idx(i, j, k)] += corpus.y_start_index;
        }

        for(size_t j = 0; j < k; j++) {
            size_t jth_nn = result.nidx[idx(i, j, k)] - corpus.y_start_index;

            result.ndist[idx(i, j, k)] = D[idx(i, jth_nn, corpus.n_packet)];
        }
    }

    return result;
}

// it is assumed that the p1.y_end_index == p2.y_start_index
// for example we combine the k nearest neighbors of x[0:100] in both results
// but the first is the k nearest neighbors from y[0:100] and the second is the k nearest neighbors from y[100:200]
ResultPacket combineKnnResultsSameX(const ResultPacket& p1, const ResultPacket& p2) {

    if(!(p1.x_end_index == p2.x_start_index)) {
        throw std::runtime_error("Cannot combine knn results, y indices are not contiguous");
    }

    if(!ResultPacket::xCombinable(p1, p2)) {
        throw std::runtime_error("Cannot combine knn results");
    }

    ResultPacket result(
        p1.m_packet, p1.k,
        p1.x_start_index, p2.x_end_index,
        p1.y_start_index, p2.y_end_index
    );

    for(size_t i = 0; i < result.m_packet; i++) {
        size_t l_idx = 0, r_idx = 0;

        while(l_idx + r_idx < result.k) {
            double l_dist = p1.ndist[idx(i, l_idx, p1.k)];
            double r_dist = p2.ndist[idx(i, r_idx, p2.k)];

            if (l_dist < r_dist) {
                result.ndist[idx(i, l_idx + r_idx, result.k)] = l_dist;
                result.nidx[idx(i, l_idx + r_idx, result.k)] = p1.nidx[idx(i, l_idx, p1.k)];
                l_idx++;
            } else {
                result.ndist[idx(i, l_idx + r_idx, result.k)] = r_dist;
                result.nidx[idx(i, l_idx + r_idx, result.k)] = p2.nidx[idx(i, r_idx, p2.k)];
                r_idx++;
            }
        }
    }

    return result;
}

// it is assumed that p1.x_end_index == p2.x_start_index
// for example we combine the k nearest neighbors from y[0:100] in both results
// the first is the k nearest neighbors of x[0:100] and the second is the k nearest neighbors of x[100:200]
ResultPacket combineKnnResultsSameY(const ResultPacket& p1, const ResultPacket& p2) {
    if(!(p1.y_end_index == p2.y_start_index)) {
        throw std::runtime_error("Cannot combine knn results, y indices are not contiguous");
    }

    if(!ResultPacket::yCombinable(p1, p2)) {
        throw std::runtime_error("Cannot combine knn results");
    }

    ResultPacket result(
        p1.m_packet + p2.m_packet, p1.k,
        p1.x_start_index, p2.x_end_index,
        p1.y_start_index, p1.y_end_index
    );

    for(size_t i = 0; i < p1.m_packet; i++) {
        for(size_t j = 0; j < p1.k; j++) {
            result.ndist[idx(i, j, p1.k)] = p1.ndist[idx(i, j, p1.k)];
            result.nidx[idx(i, j, p1.k)] = p1.nidx[idx(i, j, p1.k)];
        }
    }

    for(size_t i = 0; i < p2.m_packet; i++) {
        for(size_t j = 0; j < p2.k; j++) {
            result.ndist[idx(i + p1.m_packet, j, p2.k)] = p2.ndist[idx(i, j, p2.k)];
            result.nidx[idx(i + p1.m_packet, j, p2.k)] = p2.nidx[idx(i, j, p2.k)];
        }
    }

    return result;
} // maybe move one of the ndist and nidx vectors to the other one
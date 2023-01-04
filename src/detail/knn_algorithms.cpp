#include "knn_structs.hpp"
#include "knn_utils.hpp"

#include <cblas.h>
#include <numeric>
#include <algorithm>

// #define SELF_NOT_IN_KNN

ResultPacket knn_simple(const QueryPacket &query, const CorpusPacket &corpus,
                        size_t k_arg)
{
    // Initialize result packet with appropriate metadata
    ResultPacket results(query.m_packet, corpus.n_packet,
                         std::min(k_arg, corpus.n_packet), query.x_start_index,
                         query.x_end_index, corpus.y_start_index,
                         corpus.y_end_index);

    // Calculate kNN in the dumbest but easiest way
    struct index_distance_pair
    {
        double distance;
        size_t index;

        bool operator<(index_distance_pair const &rh) const
        {
            return distance < rh.distance;
        }
    };

    size_t k = results.k;
    size_t d = query.d;

    results.nidx.resize(query.m_packet * k);
    results.ndist.resize(query.m_packet * k);

    std::vector<index_distance_pair> idx_dist_vec(corpus.n_packet);
    for (unsigned int x = 0; x < query.m_packet; x++)
    {
        for (unsigned int y = 0; y < corpus.n_packet; y++)
        {
            double distance = 0;
            for (unsigned int i = 0; i < d; i++)
            {
                distance += (query.X[d * x + i] - corpus.Y[d * y + i]) *
                            (query.X[d * x + i] - corpus.Y[d * y + i]);
            }

#ifdef SELF_NOT_IN_KNN
            if(query.x_start_index + x == corpus.y_start_index + y){
                distance = std::numeric_limits<double>::max();
            }     
#endif

            idx_dist_vec[y] = {distance, y + corpus.y_start_index};
        }

        std::partial_sort(idx_dist_vec.begin(), idx_dist_vec.begin() + k,
                          idx_dist_vec.end());
        // write result
        for (unsigned int i = 0; i < k; i++)
        {
            results.nidx[idx(x, i, k)] = idx_dist_vec[i].index;
            results.ndist[idx(x, i, k)] = idx_dist_vec[i].distance;
        }
    }

    return results;
}

ResultPacket knn_blas(const QueryPacket &query,
                      const CorpusPacket &corpus, size_t k_arg)
{

    // Initialize result packet with appropriate metadata
    ResultPacket res(query.m_packet, corpus.n_packet,
                     std::min(k_arg, corpus.n_packet), query.x_start_index,
                     query.x_end_index, corpus.y_start_index,
                     corpus.y_end_index);

    const size_t d = query.d;
    const size_t k = res.k;

    std::vector<double> X2(res.m_packet);
    for (size_t i = 0; i < res.m_packet; i++)
    {
        X2[i] = cblas_ddot(
            d,
            &query.X[idx(i, 0, query.d)], 1,
            &query.X[idx(i, 0, query.d)], 1
        );
    }

    std::vector<double> Y2(corpus.n_packet);
    for (size_t i = 0; i < corpus.n_packet; i++)
    {
        Y2[i] = cblas_ddot(
            d,
            &corpus.Y[idx(i, 0, d)], 1,
            &corpus.Y[idx(i, 0, d)], 1
        );
    }

    std::vector<double> D(query.m_packet * corpus.n_packet);

    // Dij = x_i^2 + y_j^2
    for (size_t i = 0; i < query.m_packet; i++)
    {
        // maybe cache X2_i
        // X2_i = X2[i];
        for (size_t j = 0; j < corpus.n_packet; j++)
        {
            D[idx(i, j, corpus.n_packet)] = X2[i] + Y2[j];
        }
    }

    // Dij += -2 * x_i * y_j
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, query.m_packet,
                corpus.n_packet, corpus.d, -2.0, query.X.data(), query.d,
                corpus.Y.data(), corpus.d, 1.0, D.data(), corpus.n_packet);

#ifdef SELF_NOT_IN_KNN
    // they are overlapping
        if(corpus.y_start_index < query.x_end_index&& query.x_start_index < corpus.y_end_index) {
            for(size_t i = 0; i < query.m_packet; i++) {
                for(size_t j = 0; j < corpus.n_packet; j++) {
                    if(query.x_start_index + i == corpus.y_start_index + j)
                        D[idx(i, j, corpus.n_packet)] = std::numeric_limits<double>::max();
                }
            }
        }
#endif

/*
        auto ys = corpus.y_start_index;
        auto ye = corpus.y_end_index;
        auto xs = query.x_start_index;
        auto xe = query.x_end_index;

        if(ys > xe || xs > ye) {
            // no overlap
        } else {
            // overlap
            auto os = std::max(xs, ys);
            auto oe = std::min(xe, ye);

            if(xe - xs < ys - ye) {
                // x is shorter
                for(size_t )
            } else {
                // y is shorter
            }
        }
*/

    res.nidx.resize(query.m_packet * k);
    res.ndist.resize(query.m_packet * k);

    for (size_t i = 0; i < query.m_packet; i++)
    {
        std::vector<size_t> indices(corpus.n_packet);
        std::iota(indices.begin(), indices.end(), 0);

        size_t n_packet = corpus.n_packet;

        std::partial_sort_copy(
            indices.begin(), indices.end(), &res.nidx[idx(i, 0, k)],
            &res.nidx[idx(i, k, k)], [&D, i, n_packet](size_t a, size_t b)
            { return D[idx(i, a, n_packet)] < D[idx(i, b, n_packet)]; });

        // convert y index to global y index
        for (size_t j = 0; j < k; j++)
        {
            res.nidx[idx(i, j, k)] += corpus.y_start_index;
        }

        for (size_t j = 0; j < k; j++)
        {
            size_t jth_nn = res.nidx[idx(i, j, k)] - corpus.y_start_index;

            res.ndist[idx(i, j, k)] = D[idx(i, jth_nn, corpus.n_packet)];
        }
    }

    return res; // copy elision hopefully
}

ResultPacket knn_dynamic(const QueryPacket &query, const CorpusPacket &corpus, size_t k_arg)
{

    struct index_distance_pair
    {
        // Pair index & distance in a class, so we can easily
        // store them in a data structure and sort them based on distance.

        double distance;
        size_t index;

        bool operator<(index_distance_pair const &rh) const
        {
            return distance < rh.distance;
        }
    };

    // Make packet with appropriate metadata
    ResultPacket res(query.m_packet, corpus.n_packet, std::min(k_arg, corpus.n_packet),
                     query.x_start_index, query.x_end_index, corpus.y_start_index, corpus.y_end_index);

    size_t k = res.k;
    size_t d = query.d;

    // A bounded max heap is used to store idx-distance of only k nearest points
    size_t max_heap_size = k;
    std::vector<index_distance_pair> heap;
    heap.reserve(k);
    for (unsigned int x = 0; x < query.m_packet; x++)
    {
        heap.clear();
        for (unsigned int y = 0; y < corpus.n_packet; y++)
        {
            double distance = 0;
            for (unsigned int i = 0; i < d; i++)
            {
                distance += (query.X[d * x + i] - corpus.Y[d * y + i]) *
                            (query.X[d * x + i] - corpus.Y[d * y + i]);
            }
            
#ifdef SELF_NOT_IN_KNN
                if(query.x_start_index + x == corpus.y_start_index + y){
                    distance = std::numeric_limits<double>::max();
                }
#endif
            size_t global_y_idx = y+res.y_start_index;
            if (heap.size() < max_heap_size){
                heap.push_back(index_distance_pair{distance, global_y_idx});
                std::make_heap(heap.begin(), heap.end());
            }
            else if(distance < heap[0].distance)
            {
                heap[0] = index_distance_pair{distance, global_y_idx};
                std::make_heap(heap.begin(), heap.end());
            }
        }    

        res.nidx.resize(query.m_packet * k);
        res.ndist.resize(query.m_packet * k);
        // write result from heap to vectors
        for (unsigned int i = 0; i < k; i++)
        {
            res.nidx[idx(x, i, k)] = heap[i].index;
            res.ndist[idx(x, i, k)] = heap[i].distance;
        }
    }
    return res;
}
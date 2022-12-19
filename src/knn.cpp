#include "knn.hpp"

#include <vector>
#include <algorithm>
#include <chrono>
#include <numeric>

#include <cblas.h>
#define idx(i, j, ld) (((i)*(ld))+(j))

#define ROWMAJOR 1

#define max_block_size 10000

struct knnWorkload {
    const std::vector<double>& X;
    size_t x_start_idx;
    size_t x_end_idx; 
    size_t m_batch; //!< Number of query points in this batch    [scalar]

    const std::vector<double>& Y;
    size_t y_start_idx;
    size_t y_end_idx;
    size_t n_batch; //!< Number of corpus points in this batch   [scalar]

    size_t d; //!< Dimension of the points                 [scalar]
    size_t k; //!< Number of nearest neighbors             [scalar]
};

struct knnWorkloadResult {
    size_t x_start_idx;
    size_t x_end_idx; 

    size_t y_start_idx;
    size_t y_end_idx;

    // global indices of nearest neighbors [m_batch-by-k]
    std::vector<size_t> nidx;

    // Distance of neighbors in the same order [m_batch-by-k]
    // In global indices:
    // ndist[idx(i, j, k)] is the distance between X[i + x_start_idx, :] and Y[nidx(idx(i, j, k), :]
    std::vector<double> ndist;

    size_t m_batch; //!< Number of query points in this batch    [scalar]
    size_t n_batch; //!< Number of corpus points in this batch   [scalar]
    size_t d; //!< Dimension of the points                 [scalar]
    size_t k; //!< Number of nearest neighbors             [scalar]

    // empty constructor
    knnWorkloadResult() {
        x_start_idx = 0;
        x_end_idx = 0;
        y_start_idx = 0;
        y_end_idx = 0;
        m_batch = 0;
        n_batch = 0;
        d = 0;
        k = 0;
    }

    // move constructor taking ownership of the vectors passed
    knnWorkloadResult(
        size_t x_start_idx,
        size_t x_end_idx,
        size_t y_start_idx,
        size_t y_end_idx,
        std::vector<size_t>&& nidx,
        std::vector<double>&& ndist,
        size_t m_batch,
        size_t n_batch,
        size_t d,
        size_t k
    ) : x_start_idx(x_start_idx),
        x_end_idx(x_end_idx),
        y_start_idx(y_start_idx),
        y_end_idx(y_end_idx),
        nidx(std::move(nidx)),
        ndist(std::move(ndist)),
        m_batch(m_batch),
        n_batch(n_batch),
        d(d),
        k(k) {}


    // std::move operator
    knnWorkloadResult& operator=(knnWorkloadResult&& other) {
        x_start_idx = other.x_start_idx;
        x_end_idx = other.x_end_idx;
        y_start_idx = other.y_start_idx;
        y_end_idx = other.y_end_idx;
        nidx = std::move(other.nidx);
        ndist = std::move(other.ndist);
        m_batch = other.m_batch;
        n_batch = other.n_batch;
        d = other.d;
        k = other.k;
        return *this;
    }
};

knnWorkloadResult knnDistributed(knnWorkload wl) {
    size_t m_batch = wl.m_batch;
    size_t n_batch = wl.n_batch;
    size_t d = wl.d;
    size_t k = wl.k;

    std::vector<double> X2(m_batch);
    for(size_t i = 0; i < m_batch; i++) {
        X2[i] = cblas_ddot(
            d, 
            &wl.X[idx(i, wl.x_start_idx, d)], 1,
            &wl.X[idx(i, wl.x_start_idx, d)], 1
        );
    }

    std::vector<double> Y2(n_batch);
    for(size_t i = 0; i < n_batch; i++) {
        Y2[i] = cblas_ddot(
            d,
            &wl.Y[idx(i, wl.y_start_idx, d)], 1,
            &wl.Y[idx(i, wl.y_start_idx, d)], 1
        );
    }

    std::vector<double> D(m_batch * n_batch);

    // m, n, d refer to Y after transposition
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        m_batch, n_batch, d, 
        -2.0, 
        &wl.X[wl.x_start_idx], d, 
        &wl.Y[wl.y_start_idx], d, 
        0.0,
        D.data(), wl.n_batch
    );

    for(size_t i = 0; i < m_batch; i++)
        for(size_t j = 0; j < n_batch; j++)
            // D[idx(i, j, wl.n_batch)] is the distance between X[i + x_start_idx,:] and Y[j + y_start_idx,:]
            D[idx(i, j, n_batch)] += X2[i + wl.x_start_idx] + Y2[j + wl.y_start_idx];

    std::vector<size_t> nidx(m_batch * k);
    std::vector<double> ndist(m_batch * k);

    std::vector<size_t> indices(n_batch);

    for(size_t i = 0; i < m_batch; i++) {
        std::iota(indices.begin(), indices.end(), wl.y_start_idx);

        std::partial_sort_copy(
            indices.begin(), indices.end(),
            &nidx[idx(i, 0, d)], &nidx[idx(i, k, d)],
            [&D, n_batch, i](size_t a, size_t b) {
                return D[idx(i, a, n_batch)] < D[idx(i, b, n_batch)];
            }
        );

        for(size_t j = 0; j < k; j++) {
            ndist[idx(i, j, k)] = D[idx(i, nidx[j] - wl.y_start_idx, n_batch)];
        }
    }

    return {
        wl.x_start_idx, wl.x_end_idx, wl.y_start_idx, wl.y_end_idx,
        std::move(nidx), std::move(ndist),
        wl.m_batch, wl.n_batch, wl.d, wl.k
    };
};

// combines both into the knnresult with the largest m
// returns 0 if that is a and b otherwise
void combineKnnresultsHorizontal(knnWorkloadResult& left, knnWorkloadResult& right){

    // starting cold
    if (left.k == 0 && left.d == 0 && left.m_batch == 0 && left.n_batch == 0) {
        left = std::move(right);
        return;
    }

    if (left.k != right.k) {
        throw std::runtime_error("k must be the same for both knnWorkloadResults");
    } if (left.d != right.d) {
        throw std::runtime_error("d must be the same for both knnWorkloadResults");
    } if (left.m_batch != right.m_batch) {
        throw std::runtime_error("n must be the same for both knnWorkloadResults");
    } if (left.x_start_idx != right.x_start_idx || left.x_end_idx != right.x_end_idx) {
        throw std::runtime_error("x_start_idx and y_end_idx must be the same for both knnWorkloadResults");
    } if (left.y_end_idx != right.y_start_idx) {
        throw std::runtime_error("lefts's y_end_idx must be the same as rights's y_start_idx");
    }

    left.n_batch += right.n_batch;
    left.y_end_idx = right.y_end_idx;

    std::vector<size_t> nidx(left.m_batch * left.k);
    std::vector<double> ndist(left.m_batch * left.k);

    for(size_t i = 0; i < left.m_batch; i++) {
        size_t l_ptr = 0, r_ptr = 0;

        while (l_ptr + r_ptr < left.k) {
            size_t l_idx = left.nidx[idx(i, l_ptr, left.d)];
            size_t r_idx = right.nidx[idx(i, r_ptr, right.d)];

            double l_dist = left.ndist[idx(i, l_ptr, left.d)];
            double r_dist = right.ndist[idx(i, r_ptr, right.d)];

            if (l_dist < r_dist) {
                nidx[idx(i, l_ptr + r_ptr, left.d)] = l_idx;
                ndist[idx(i, l_ptr + r_ptr, left.d)] = left.ndist[idx(i, l_ptr, left.d)];
                l_ptr++;
            } else {
                nidx[idx(i, l_ptr + r_ptr, left.d)] = r_idx;
                ndist[idx(i, l_ptr + r_ptr, left.d)] = right.ndist[idx(i, r_ptr, right.d)];
                r_ptr++;
            }
        }
    }

    // previously held stuff is auto-released
    left.nidx = std::move(nidx);
    left.ndist = std::move(ndist);

    return;
}

// result is in up, down is discarded
void combineKnnresultsVertical(knnWorkloadResult& up, knnWorkloadResult& down){
    if(up.k == 0 && up.d == 0 && up.m_batch == 0 && up.n_batch == 0) {
        up = std::move(down);
        return;
    }


    if (up.k != down.k) {
        throw std::runtime_error("k must be the same for both knnWorkloadResults");
    } if (up.d != down.d) {
        throw std::runtime_error("d must be the same for both knnWorkloadResults");
    } if (up.n_batch != down.n_batch) {
        throw std::runtime_error("n must be the same for both knnWorkloadResults");
    } if (up.y_start_idx != down.y_start_idx || up.y_end_idx != down.y_end_idx) {
        throw std::runtime_error("y_start_idx and y_end_idx must be the same for both knnWorkloadResults");
    } if (up.x_end_idx != down.x_start_idx) {
        throw std::runtime_error("up's x_end_idx must be the same as down's x_start_idx");
    }

    up.x_end_idx = down.x_end_idx;
    up.m_batch += down.m_batch;

    up.nidx.insert(up.nidx.end(), down.nidx.begin(), down.nidx.end());
    up.ndist.insert(up.ndist.end(), down.ndist.begin(), down.ndist.end());

    return;
}

//! Compute k nearest neighbors of each point in Y [m-by-d]
/*!
\param X Query data points            [n-by-d]
\param Y Corpus data points           [m-by-d]
\param m Number of query points       [scalar]
\param n Number of corpus posize_ts   [scalar]
\param d Number of dimensions         [scalar]
\param k Number of neighbors          [scalar]

\return the kNN result
*/
knnresult kNN(const std::vector<double>& X, const std::vector<double>& Y, const size_t m, const size_t n, const size_t d, size_t k)
{
    /*
    if(magic){
        std::cout << "Using magic" << std::endl;
    } else {
        std::cout << "Using not magic" << std::endl;
    }
    */

    if (k > n) {
        k = n;
    }

    size_t num_workers = 3;
    size_t m_per_batch = m / num_workers;
    size_t n_per_batch = n / num_workers;

    knnWorkloadResult result;
    for(size_t hor_block = 0; hor_block < num_workers; hor_block++) {
        size_t x_start_idx = hor_block * m_per_batch;
        size_t x_end_idx = hor_block == num_workers-1 ? m : (hor_block + 1) * m_per_batch;

        knnWorkloadResult superblock_result;
        for(size_t ver_block = 0; ver_block < num_workers; ver_block++) {
            size_t y_start_idx = ver_block * n_per_batch;
            size_t y_end_idx = ver_block == num_workers-1 ? n : (ver_block + 1) * n_per_batch;

            if (ver_block == num_workers - 1) {
                y_end_idx = n;
            }

            knnWorkload wl = { 
                X, x_start_idx, x_end_idx, m,
                Y, y_start_idx, y_end_idx, n, 
                d, k
            };

            knnWorkloadResult block_result = knnDistributed(wl);
            combineKnnresultsHorizontal(superblock_result, block_result);
            
            std::cout << "x_start_idx: " << result.x_start_idx << " x_end_idx: " << result.x_end_idx << " y_start_idx: " << result.y_start_idx << " y_end_idx: " << result.y_end_idx << std::endl;
        }

        combineKnnresultsVertical(result, superblock_result);
    }

    return {
        std::move(result.nidx), std::move(result.ndist),
        result.m_batch, result.k
    };
}


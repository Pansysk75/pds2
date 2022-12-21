#include "knn.hpp"

#include <vector>
#include <algorithm>
#include <chrono>
#include <numeric>

#include <cblas.h>

#define idx(i, j, ld) (((i)*(ld))+(j))
#define DEB(x) if(debug) std::cout << x << std::endl;

struct knnWorkload {
    const std::vector<double>& X;
    // start and end indices are not multiplied by d
    size_t x_start_idx;
    size_t x_end_idx; 
    size_t m_batch; //!< Number of query points in this batch    [scalar]

    const std::vector<double>& Y;
    size_t y_start_idx;
    size_t y_end_idx;
    size_t n_batch; //!< Number of corpus points in this batch   [scalar]

    size_t d; //!< Dimension of the points                 [scalar]
    size_t k; //!< Number of nearest neighbors             [scalar]

    knnWorkload(const std::vector<double>& X, size_t x_start_idx, size_t x_end_idx, 
                const std::vector<double>& Y, size_t y_start_idx, size_t y_end_idx, 
                size_t d, size_t k) : 
                X(X), x_start_idx(x_start_idx), x_end_idx(x_end_idx),
                Y(Y), y_start_idx(y_start_idx), y_end_idx(y_end_idx),
                d(d), k(k) 
                {
        m_batch = x_end_idx - x_start_idx;
        n_batch = y_end_idx - y_start_idx;
    }

    std::string to_string() const {
        return "x_start_idx = " + std::to_string(x_start_idx) + "\n" +
                "x_end_idx = " + std::to_string(x_end_idx) + "\n" +
                "y_start_idx = " + std::to_string(y_start_idx) + "\n" +
                "y_end_idx = " + std::to_string(y_end_idx) + "\n" +
                "m_batch = " + std::to_string(m_batch) + "\n" +
                "n_batch = " + std::to_string(n_batch) + "\n" +
                "d = " + std::to_string(d) + "\n" +
                "k = " + std::to_string(k) + "\n";
    }
};

struct knnWorkloadResult {
    // start and end indices are multiplied by d
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
        nidx = std::vector<size_t>();
        ndist = std::vector<double>();
        d = 0;
        k = 0;
    }

    // constructor that moves vectors in
    knnWorkloadResult(size_t x_start_idx, size_t x_end_idx, size_t y_start_idx, size_t y_end_idx, std::vector<size_t>&& nidx, std::vector<double>&& ndist, size_t m_batch, size_t n_batch, size_t d, size_t k) {
        this->x_start_idx = x_start_idx;
        this->x_end_idx = x_end_idx;
        this->y_start_idx = y_start_idx;
        this->y_end_idx = y_end_idx;

        this->nidx = std::move(nidx);
        this->ndist = std::move(ndist);
        nidx = std::vector<size_t>();
        ndist = std::vector<double>();

        this->m_batch = m_batch;
        this->n_batch = n_batch;
        this->d = d;
        this->k = k;
    }

    // move constructor
    knnWorkloadResult(knnWorkloadResult&& other) {
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

        other.x_start_idx = 0;
        other.x_end_idx = 0;
        other.y_start_idx = 0;
        other.y_end_idx = 0;
        other.m_batch = 0;
        other.n_batch = 0;
        other.d = 0;
        other.k = 0;
    }

    knnWorkloadResult& operator=(knnWorkloadResult&& other) {
        if(this == &other){
            return *this;
            DEB("Self assignment");
        }

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

        other.x_start_idx = 0;
        other.x_end_idx = 0;
        other.y_start_idx = 0;
        other.y_end_idx = 0;
        other.nidx = std::vector<size_t>();
        other.ndist = std::vector<double>();
        other.m_batch = 0;
        other.n_batch = 0;

        return *this;
    }

    std::string to_string() const {
        DEB("knnWorkloadResult to_string called");
        return "x_start_idx = " + std::to_string(x_start_idx) + "\n" +
                "x_end_idx = " + std::to_string(x_end_idx) + "\n" +
                "y_start_idx = " + std::to_string(y_start_idx) + "\n" +
                "y_end_idx = " + std::to_string(y_end_idx) + "\n" +
                "m_batch = " + std::to_string(m_batch) + "\n" +
                "n_batch = " + std::to_string(n_batch) + "\n" +
                "d = " + std::to_string(d) + "\n" +
                "k = " + std::to_string(k) + "\n";
    }
};

knnWorkloadResult knnBlock(const knnWorkload& wl) {
    DEB("Entering knnBlock with: ")
    DEB(wl.to_string())

    size_t m_batch = wl.m_batch;
    size_t n_batch = wl.n_batch;
    size_t d = wl.d;
    size_t k = wl.k;

    std::vector<double> X2(m_batch);
    for(size_t i = wl.x_start_idx; i < wl.x_end_idx; i++) {
        X2[i-wl.x_start_idx] = cblas_ddot(
            d,
            &wl.X[idx(i, 0, d)], 1,
            &wl.X[idx(i, 0, d)], 1
        );
    }

    std::vector<double> Y2(n_batch);
    for(size_t i = wl.y_start_idx; i < wl.y_end_idx; i++) {
        Y2[i-wl.y_start_idx] = cblas_ddot(
            d,
            &wl.Y[idx(i, 0, d)], 1,
            &wl.Y[idx(i, 0, d)], 1
        );
    }

    std::vector<double> D(m_batch * n_batch);

    // m, n, d refer to Y after transposition
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans, CblasTrans,
        m_batch, n_batch, d,
        -2.0,
        &wl.X[idx(wl.x_start_idx, 0, d)], d,
        &wl.Y[idx(wl.y_start_idx, 0, d)], d,
        0,
        D.data(), wl.n_batch
    );

    for(size_t i = 0; i < m_batch; i++)
        for(size_t j = 0; j < n_batch; j++)
            // D[idx(i, j, wl.n_batch)] is the distance between X[i + x_start_idx,:] and Y[j + y_start_idx,:]
            D[idx(i, j, n_batch)] += X2[i] + Y2[j];

    std::vector<size_t> nidx(m_batch * k);
    std::vector<double> ndist(m_batch * k);


    for(size_t i = 0; i < m_batch; i++) {
        std::vector<size_t> indices(n_batch);
        std::iota(indices.begin(), indices.end(), wl.y_start_idx);
        size_t y_start_idx = wl.y_start_idx;

        std::partial_sort_copy(
            indices.begin(), indices.end(),
            &nidx[idx(i, 0, k)], &nidx[idx(i, k, k)],
            [&D, n_batch, i, y_start_idx](size_t a, size_t b) {
                return D[idx(i, a - y_start_idx, n_batch)] < D[idx(i, b - y_start_idx, n_batch)];
            }
        );

        for(size_t j = 0; j < k; j++) {
            ndist[idx(i, j, k)] = D[idx(i, nidx[idx(i, j, k)] - wl.y_start_idx, n_batch)];
        }
    }
    X2 = std::vector<double>();

    Y2 = std::vector<double>();

    D = std::vector<double>();

    DEB("Finished knnBlock")

    return {
        wl.x_start_idx,
        wl.x_end_idx,
        wl.y_start_idx,
        wl.y_end_idx,
        std::move(nidx),
        std::move(ndist),
        m_batch,
        n_batch,
        d,
        k
    };

};

// combines both into the knnresult with the largest m
// returns 0 if that is a and b otherwise
void combineKnnresultsHorizontal(knnWorkloadResult& left, knnWorkloadResult& right){
    DEB("Starting combineKnnresultsHorizontal with:")
    DEB(left.to_string())
    DEB(right.to_string())

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
            size_t l_idx = left.nidx[idx(i, l_ptr, left.k)];
            size_t r_idx = right.nidx[idx(i, r_ptr, right.k)];

            double l_dist = left.ndist[idx(i, l_ptr, left.k)];
            double r_dist = right.ndist[idx(i, r_ptr, right.k)];

            if (l_dist < r_dist) {
                nidx[idx(i, l_ptr + r_ptr, left.k)] = l_idx;
                ndist[idx(i, l_ptr + r_ptr, left.k)] = left.ndist[idx(i, l_ptr, left.k)];
                l_ptr++;
            } else {
                nidx[idx(i, l_ptr + r_ptr, left.k)] = r_idx;
                ndist[idx(i, l_ptr + r_ptr, left.k)] = right.ndist[idx(i, r_ptr, right.k)];
                r_ptr++;
            }
        }
    }

    DEB("Moving nidx and ndist")
    // previously held stuff is auto-released
    left.nidx = std::move(nidx);
    left.ndist = std::move(ndist);

    nidx = std::vector<size_t>();
    ndist = std::vector<double>();
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
knnresult knnSerial(const std::vector<double>& X, const std::vector<double>& Y, const size_t m, const size_t n, const size_t d, size_t k)
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

    size_t m_per_batch = m / num_batches;
    size_t n_per_batch = n / num_batches;

    knnWorkloadResult result;
    for(size_t hor_block = 0; hor_block < num_batches; hor_block++) {
        size_t x_start_idx = hor_block * m_per_batch;
        size_t x_end_idx = (hor_block == num_batches-1 ? m : (hor_block + 1) * m_per_batch);

        knnWorkloadResult superblock_result;
        for(size_t ver_block = 0; ver_block < num_batches; ver_block++) {
            size_t y_start_idx = ver_block * n_per_batch;
            size_t y_end_idx = (ver_block == num_batches-1 ? n : (ver_block + 1) * n_per_batch);

            knnWorkload wl(
                X, x_start_idx, x_end_idx,
                Y, y_start_idx, y_end_idx, 
                d, k
            );

            knnWorkloadResult block_result = knnBlock(wl);
            combineKnnresultsHorizontal(superblock_result, block_result);
            
            DEB("x_start_idx: " << result.x_start_idx << " x_end_idx: " << 
            result.x_end_idx << " y_start_idx: " << result.y_start_idx << " y_end_idx: " << result.y_end_idx << std::endl);
        }

        combineKnnresultsVertical(result, superblock_result);
    }

    return {
        std::move(result.nidx), std::move(result.ndist),
        result.m_batch, result.k
    };
}
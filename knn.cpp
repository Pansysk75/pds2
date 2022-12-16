#include "knn.hpp"
#include <vector>
#include <algorithm>
#include <chrono>

#include <cblas.h>
#define idx(i, j, ld) (((i)*(ld))+(j))

#define ROWMAJOR 1


/*
struct knnresult {
    const size_t x_start_idx;
    const size_t x_end_idx;

    const size_t y_start_idx;
    const size_t y_end_idx;

    std::vector<size_t> nidx; //!< global Indices of nearest neighbors [m-by-k]
    std::vector<double> ndist; //!< Distance of neighbors in the same order [m-by-k]
    size_t m; //!< Number of query points                  [scalar]
    size_t k; //!< Number of nearest neighbors             [scalar]
}
*/

//! Compute k nearest neighbors of each point in Y [m-by-d]
/*!
\param X Query data points            [n-by-d]
\param Y Corpus data points             [m-by-d]
\param m Number of query points         [scalar]
\param n Number of corpus posize_ts        [scalar]
\param d Number of dimensions          [scalar]
\param k Number of neighbors           [scalar]

\return the kNN result
*/
knnresult kNN(const std::vector<double>& X, const std::vector<double>& Y, const size_t m, const size_t n, const size_t d, size_t k)
{
    if (k > n)
    {
        k = n;
    }

    std::vector<double> X2(m);
    std::vector<double> Y2(n);

    auto start = std::chrono::high_resolution_clock::now();
    // Compute X2 and Y2
    for (size_t i = 0; i < m; i++)
    {
        if (magic) {
            X2[i] = 0;
            for (size_t j = 0; j < d; j++)
            {
                X2[i] += X[idx(i, j, d)] * X[idx(i, j, d)];
            }
        }
        X2[i] = cblas_ddot(d, &X[idx(i, 0, d)], 1, &X[idx(i, 0, d)], 1);
    }

    for (size_t i = 0; i < n; i++)
    {
        if (magic) {
            Y2[i] = 0;
            for (size_t j = 0; j < d; j++)
            {
                Y2[i] += Y[idx(i, j, d)] * Y[idx(i, j, d)];
            }
            Y2[i] = cblas_ddot(d, &Y[idx(i, 0, d)], 1, &Y[idx(i, 0, d)], 1);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "X2 and Y2 done, time taken:" << duration.count() << "micros" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    std::vector<double> m2_xy(m * n);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                    m, n, d, -2, X.data(),
                    d, Y.data(), d, 0,
                    m2_xy.data(), n);
    end = std::chrono::high_resolution_clock::now(); 
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "m2_xy done, time taken:" << duration.count() << "micros" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    std::vector<double> dist(m*n);
    for(size_t i = 0; i < m; i++){
        for(size_t j = 0; j < n; j++){
            dist[idx(i, j, n)] = X2[i] + Y2[j] + m2_xy[idx(i, j, n)];
        }
    }
    end = std::chrono::high_resolution_clock::now(); 
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "dist done, time taken:" << duration.count() << "micros" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    std::vector<double> ndist(m*k);
    std::vector<size_t> nidx(m*k);

    // populate result.ndist and result.nidx with the indices of the k nearest neighbors
    for (size_t i = 0; i < m; i++) {
        std::vector<double> el_indices(n);

        for (size_t j = 0; j < n; j++) {
            el_indices[j] = j;
        }

        std::partial_sort(el_indices.begin(), el_indices.begin() + k, el_indices.end(), [&dist, i, n](size_t a, size_t b) {
            return dist[idx(i, a, n)] < dist[idx(i, b, n)];
        });

        for (size_t j = 0; j < k; j++) {
            nidx[idx(i, j, k)] = el_indices[j];
            ndist[idx(i, j, k)] = dist[idx(i, el_indices[j], n)];
        }
    }
    end = std::chrono::high_resolution_clock::now(); 
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "all done, time taken:" << duration.count() << "micros" << std::endl;

    //return { std::move(nidx), std::move(ndist), m, k };
    return { nidx, ndist, m, k };
}


// combines both into the knnresult with the largest m
// returns 0 if that is a and b otherwise
size_t combineKnnresults(knnresult& a, knnresult& b){
    const size_t k = a.k;

    knnresult& resultor = (a.m >= b.m) ? a : b;
    knnresult& other = (a.m >= b.m) ? b : a;

    const size_t choice = (a.m >= b.m) ? 0 : 1;

    // we want to merge them. It is assumed that the k neighbors are sorted.
    size_t r_ptr = 0, o_ptr = 0;



    return choice;
}
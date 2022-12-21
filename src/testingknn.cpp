#include <random>
#include <chrono>

#include "knn.hpp"
#include "testingknn.hpp"
#include "fileio.hpp"

#define idx(i, j, ld) (((i)*(ld))+(j))

// RowMajor 3d grid up to SxSxS
// m is query points
knnTestData regual_grid(size_t s, size_t d, size_t m) {
    // Corpus points
    size_t n = 1;
    for(size_t _ = 0; _ < d; _++) {
        n *= s;
    }

    std::vector<double> Y(n*d);

    for(size_t id = 0; id < n; id++) {
        size_t i = id;
        for(size_t comp = 0; comp < d; comp++) {
            Y[idx(id, comp, d)] = i % s;
            i /= s;
        }
    }

    // Query points, m is given
    std::vector<double> X(m*d);

    // random points in the grid
    for(size_t i = 0; i < m; i++){
        const size_t y_choice = rand() % n;
        for(size_t comp = 0; comp < d; comp++)
            X[idx(i, comp, d)] = Y[idx(y_choice, comp, d)];
    }

    // 3**d is the imediate neighbors
    size_t k = 1;
    for(size_t i = 0; i < d; i++){
        k *= 3;
    }
    return { std::move(X), std::move(Y), m, n, d, k };
}

knnTestData random_grid(size_t m, size_t n, size_t d, size_t k) {
    size_t s = 1000;

    std::vector<double> Y(n*d);

    for(size_t id = 0; id < n; id++) {
        for(size_t comp = 0; comp < d; comp++) {
            Y[idx(id, comp, d)] = rand() % s;
        }
    }

    // Query points, m is given
    std::vector<double> X(m*d);

    // random points in the grid
    for(size_t i = 0; i < m; i++){
        const size_t y_choice = rand() % n;
        for(size_t comp = 0; comp < d; comp++)
            X[idx(i, comp, d)] = Y[idx(y_choice, comp, d)];
    }

    return { std::move(X), std::move(Y), m, n, d, k };
}

knnresult runData(const std::vector<double>& X, const std::vector<double>& Y, const size_t m, const size_t n, const size_t d, const size_t k) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t _k = k;
    knnresult result = knnSerial(X, Y, m, n, d, _k);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;

    return result;
}

knnresult runData(const knnTestData& data) {
    return runData(data.X, data.Y, data.m, data.n, data.d, data.k);
}

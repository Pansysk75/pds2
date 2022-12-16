#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <string>
#include <cblas.h>
#include <random>

#include "knn.hpp"
#include "fileio.hpp"

#define idx(i, j, ld) (((i)*(ld))+(j))

struct knnTestData {
    std::vector<double> X;
    std::vector<double> Y;

    size_t m;
    size_t n;
    size_t d;
    size_t k;
};

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
    knnresult result = kNN(X, Y, m, n, d, _k);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;

    return result;
}

knnresult runData(const knnTestData& data) {
    return runData(data.X, data.Y, data.m, data.n, data.d, data.k);
}

int main(int argc, char** argv)
{
    // Sample input values
    /*
    size_t d = 28*28;

    std::vector<double> X = load_csv<double>("../mnist_test.csv", std::stoi(argv[1]));
    size_t m = X.size()/d;
    std::cout << "Loaded " << m << " points from mnist_test.csv" << std::endl;

    std::vector<double> Y = load_csv<double>("../mnist_train.csv", std::stoi(argv[2]));
    size_t n = Y.size()/d;
    std::cout << "Loaded " << n << " points from mnist_train.csv" << std::endl;

    size_t k = 1;
    */

    size_t m = std::stoi(argv[1]);
    size_t n = std::stoi(argv[2]);
    size_t d = std::stoi(argv[3]);
    size_t k = std::stoi(argv[4]);

    knnTestData knnData = random_grid(m, n, d, k);

    knnresult result = runData(knnData);

    // Print the results
    for (size_t i = 0; i < std::min(result.m, (size_t)0); i++)
    {
        std::cout << "Nearest neighbors of point ";
        for (size_t j = 0; j < knnData.d; j++)
        {
            //std::cout << knnData.X[idx(i, j, knnData.d)] << " ";
        }
        std::cout << "are:" << std::endl;

        for (size_t j = 0; j < 1; j++)
        {
            double diff = 0;
            for(size_t comp = 0; comp < knnData.d; comp++){
                diff += std::abs(knnData.X[idx(i, comp, knnData.d)] - knnData.Y[idx(result.nidx[idx(i, j, result.k)],comp , knnData.d)]);
                //std::cout << knnData.Y[idx(result.nidx[idx(i, j, result.k)],comp , knnData.d)] << " ";
            }
            std::cout << "diff: " << diff << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}


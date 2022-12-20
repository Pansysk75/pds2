#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <string>
#include <cblas.h>
#include <random>

#include "global_vars.hpp"
#include "knn.hpp"
#include "testingknn.hpp"
#include "fileio.hpp"

#define idx(i, j, ld) (((i)*(ld))+(j))

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

/*
    size_t m = std::stoi(argv[1]);
    size_t n = std::stoi(argv[2]);
    size_t d = std::stoi(argv[3]);
    size_t k = std::stoi(argv[4]);

    debug = std::stoi(argv[5]) == 1;
    knnTestData knnData = random_grid(m, n, d, k);
*/

    size_t s = std::stoi(argv[1]);
    size_t d = std::stoi(argv[2]);
    size_t m = std::stoi(argv[3]);

    debug = std::stoi(argv[4]) == 1;

    knnTestData knnData = regual_grid(s, d, m);

    knnresult result = runData(knnData);

    // Print the results
    for (size_t i = 0; i < std::min(result.m, (size_t)5); i++)
    {
        std::cout << "Nearest neighbors of point ";
        for (size_t j = 0; j < knnData.d; j++)
        {
            std::cout << knnData.X[idx(i, j, knnData.d)] << " ";
        }
        std::cout << "are:" << std::endl;

        for (size_t j = 0; j < result.k; j++)
        {
            double diff = 0;
            for(size_t comp = 0; comp < knnData.d; comp++){
                //diff += std::abs(knnData.X[idx(i, comp, knnData.d)] - knnData.Y[idx(result.nidx[idx(i, j, result.k)],comp , knnData.d)]);
                std::cout << knnData.Y[idx(result.nidx[idx(i, j, result.k)],comp , knnData.d)] << " ";
            }
            //std::cout << "diff: " << diff << std::endl;
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}


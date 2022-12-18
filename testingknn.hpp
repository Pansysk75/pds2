# pragma once

#include <vector>

#include "knn.hpp"
#include "global_vars.hpp"

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
knnTestData regual_grid(size_t s, size_t d, size_t m);

knnTestData random_grid(size_t m, size_t n, size_t d, size_t k);

knnresult runData(const std::vector<double>& X, const std::vector<double>& Y, const size_t m, const size_t n, const size_t d, const size_t k);

knnresult runData(const knnTestData& data);
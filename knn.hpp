# pragma once

#include <vector>
#include <algorithm>
#include <iostream>

bool magic = true;

struct knnresult
{
    std::vector<size_t> nidx; // Indices (0-based) of nearest neighbors    [m-by-k]
    std::vector<double> ndist; // Distance of nearest neighbors         [m-by-k]
    size_t m; // Number of query points                        [scalar]
    size_t k; // Number of nearest neighbors                   [scalar]
};

//! Compute k nearest neighbors of each point in Y [m-by-d]
/*!
\param X Query data points            [n-by-d]
\param Y Corpus data points             [m-by-d]
\param m Number of query points         [scalar]
\param n Number of corpus points        [scalar]
\param d Number of dimensions          [scalar]
\param k Number of neighbors           [scalar]
\return the kNN result
*/
knnresult kNN(const std::vector<double>& X, const std::vector<double>& Y, const size_t m, const size_t n, const size_t d, size_t k);
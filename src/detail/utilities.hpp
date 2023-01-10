#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <algorithm>
#include <iomanip>

#include "knn_utils.hpp"

namespace utilities
{

	class timer
	{

	private:
		std::chrono::high_resolution_clock::time_point t1;
		std::chrono::high_resolution_clock::time_point t2;

	public:
		void start() { t1 = std::chrono::high_resolution_clock::now(); }
		void stop() { t2 = std::chrono::high_resolution_clock::now(); }
		double get() { return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count(); }
	};

}

void print_mnist(double* grid) {
  for (size_t i = 0; i < 28; ++i) {
    for (size_t j = 0; j < 28; ++j) {
        // take up 3 digits
        std::cout << std::setw(3) << (int)grid[i*28 + j] << ' ';
    }
    std::cout << '\n';
  }
}

void print_results_with_labels(const std::string& q_filename, const std::string& c_filename, const ResultPacket& result, size_t d, bool mnistPrint = false)
{
    auto [query, query_labels, corpus, corpus_labels] = file_packets_with_label(q_filename, 0, result.m_packet, c_filename, 0, result.n_packet, d);

    // Print the results
    for (size_t i = 0; i < std::min(result.m_packet, 10ul); i++)
    {
        std::cout << "The calculated " << std::min(result.k, (size_t)5ul) << " nearest neighbours of vector with label: "
                  << query_labels[i] << std::endl;
        if(mnistPrint) print_mnist(&query.X[idx(i, 0, d)]);
        std::cout << "Are the ones with labels:" << std::endl;

        for (size_t j = 0; j < std::min(result.k, (size_t)5ul); j++)
        {
            std::cout << corpus_labels[result.nidx[idx(i, j, result.k)]]
                      << " with an MSE of: " << result.ndist[idx(i, j, result.k)]
                      << std::endl;
            if (mnistPrint) print_mnist(&corpus.Y[idx(result.nidx[idx(i, j, result.k)], 0, d)]);
        }
        std::cout << std::endl;
    }
}

void print_results_with_labels(const std::string& filename, const ResultPacket &result, size_t d, bool mnistPrint)
{
    print_results_with_labels(filename, filename, result, d, mnistPrint);
}

void print_results(const std::string& q_filename, const std::string& c_filename, const ResultPacket& result, size_t d)
{
    auto [query, corpus] = file_packets(q_filename, 0, result.m_packet, c_filename, 0, result.n_packet, d);

    // Print the results
    for (size_t i = 0; i < std::min(result.m_packet, 10ul); i++)
    {
        std::cout << "The calculated " << std::min(result.k, (size_t)5ul) << " nearest neighbours of point in line: "
                  << i << std::endl;
        std::cout << "Are the points in lines:" << std::endl;

        for (size_t j = 0; j < std::min(result.k, (size_t)5ul); j++)
        {
            std::cout << result.nidx[idx(i, j, result.k)]
                      << " with an MSE of: " << result.ndist[idx(i, j, result.k)]
                      << std::endl;
        }
        std::cout << std::endl;
    }
}

void print_results(const std::string& filename, const ResultPacket &result, const size_t d)
{
    print_results(filename, filename, result, d);
}
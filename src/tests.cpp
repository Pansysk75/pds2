#include <iostream>
#include <string>
#include <vector>

#include "detail/fileio.hpp"
#include "detail/knn_algorithms.hpp"
#include "detail/knn_utils.hpp"
#include "detail/utilities.hpp"

bool compareResults(ResultPacket &p1, ResultPacket &p2) {
  bool flag = (p1.k == p2.k) && (p1.m_packet == p2.m_packet) &&
              (p1.n_packet == p2.n_packet);

  if (flag == false) {
    return false;
    std::cout << "Mistake in flags" << std::endl;
  }

  // For every point in query, if knn indices are the same
  flag = true;
  for (size_t i = 0; (i < p1.m_packet) && (flag == true); i++) {
    auto p1_begin = p1.nidx.begin() + i * p1.k;
    auto p1_end = p1.nidx.begin() + (i + 1) * p1.k;
    auto p2_begin = p2.nidx.begin() + i * p1.k;

    if (!std::is_permutation(p1_begin, p1_end, p2_begin)) {
      std::cout << "Mistake in indices" << std::endl;
      std::cout << "i = " << i << std::endl;
      for (size_t j = 0; j < p1.k; j++) {
        std::cout << p1.nidx[i * p1.k + j] << " " << p2.nidx[i * p1.k + j]
                  << std::endl;
        std::cout << p1.ndist[i * p1.k + j] << " " << p2.ndist[i * p1.k + j]
                  << std::endl;
      }
      return false;
    }
  }
  return true;
}

void test_knn(size_t size, size_t dim, size_t k, size_t idx_start,
              size_t idx_end) {

  utilities::timer timer;

  auto [query, corpus] = random_grid(size, size, dim);

  // Cool Impl
  timer.start();
  ResultPacket r2 = knn_blas(query, corpus, k);
  timer.stop();
  auto t2 = timer.get() / 1000000;

  std::cout << "Blas: " << t2 << "ms " << std::endl;

/*
  // Simple Impl
  timer.start();
  ResultPacket r1 = knn_simple(query, corpus, k);
  timer.stop();
  auto t1 = timer.get() / 1000000;

  std::cout << "Simple: " << t1 << "ms " << std::endl;

  // split for memory
  timer.start();
  ResultPacket r3 = knn_blas_in_parts(query, corpus, k, 10);
  timer.stop();
  auto t3 = timer.get() / 1000000;
  std::cout << "Blas in Parts: " << t3 << "ms " << std::endl;

  bool eq12 = compareResults(r1, r2);
  bool eq13 = compareResults(r1, r3);

  std::cout << "Completed test" << std::endl;

  std::cout << "Equality Test (1 vs 2): " << eq12 << std::endl;
  std::cout << "Equality Test (1 vs 3): " << eq13 << std::endl;
*/
}

int main(int argc, char **argv) {
  if (argc != 4 && argc != 5) {
    std::cout << "Usage: ./tests <size> <dim> <k>" << std::endl;
    return 1;
  }

  size_t size = std::stoi(argv[1]);
  size_t dim = std::stoi(argv[2]);
  size_t k = std::stoi(argv[3]);
  test_knn(size, dim, k, 0, size);

  return 0;
}

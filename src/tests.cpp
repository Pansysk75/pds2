#include <iostream>
#include <string>
#include <vector>

#include "detail/knn_algorithms.hpp"
#include "detail/utilities.hpp"
#include "detail/fileio.hpp"
#include "detail/testingknn.hpp"

bool compareResults(ResultPacket &p1, ResultPacket &p2) 
{
  bool flag =
  				(p1.k == p2.k) && 
  				(p1.m_packet == p2.m_packet) &&
              	(p1.n_packet == p2.n_packet);

  if (flag == false)
    return false;

  // For every point in query, if knn indices are the same
  flag = true;
  for (size_t i = 0; (i < p1.m_packet) && (flag == true); i++) 
  {
    auto p1_begin = p1.nidx.begin() + i * p1.k;
    auto p1_end = p1.nidx.begin() + (i + 1) * p1.k;
    auto p2_begin = p2.nidx.begin() + i * p1.k;

    flag = flag && std::is_permutation(p1_begin, p1_end, p2_begin);
  }
  return flag;
}

void test(size_t size, size_t dim, size_t k, size_t idx_start, size_t idx_end) 
{

  utilities::timer timer;

  QueryPacket query(size, dim, idx_start, idx_end);
  query.X = import_data(idx_start, idx_end, dim);

  CorpusPacket corpus(size, dim, idx_start, idx_end);
  corpus.Y = import_data(idx_start, idx_end, dim);

  // Simple Impl
  timer.start();
  ResultPacket r1 = knn_simple(query, corpus, k);
  timer.stop();
  auto t1 = timer.get() / 1000000;

  // Cool Impl
  timer.start();
  ResultPacket r2 = knn_blas(query, corpus, k);
  timer.stop();
  auto t2 = timer.get() / 1000000;

  // Whoa what an Impl
  timer.start();
  ResultPacket r3 = knn_dynamic(query, corpus, k);
  timer.stop();
  auto t3 = timer.get() / 1000000;

  // this is insane
  // that's fast
  // it's smol too
  timer.start();
  ResultPacket r4 = SyskoSimulation(query, corpus, k, 1, 10);
  timer.stop();
  auto t4 = timer.get() / 1000000;

  bool eq12 = compareResults(r1, r2);
  bool eq13 = compareResults(r1, r3);
  bool eq14 = compareResults(r1, r4);

  std::cout << "Completed test" << std::endl;
  std::cout << " t1 = " << t1 << " ms\n t2 = " << t2 << " ms\n t3 = " << t3 << " ms\n t4 = " << t4 << std::endl;

  std::cout << "Equality Test (1 vs 2): " << eq12 << std::endl;
  std::cout << "Equality Test (1 vs 3): " << eq13 << std::endl;
  std::cout << "Equality Test (1 vs 4): " << eq14 << std::endl;
}

int main() 
{

  test(20000, 2, 5, 0, 20000);

  return 0;
}

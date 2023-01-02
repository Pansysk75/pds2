#include <iostream>
#include <string>
#include <vector>

#include "detail/knn_algorithms.hpp"
#include "detail/utilities.hpp"
#include "detail/fileio.hpp"
#include "detail/testingknn.hpp"
#include "detail/mpi_process.hpp"

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

void test_knn(size_t size, size_t dim, size_t k, size_t idx_start, size_t idx_end)
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

void test_com(mpi_process &proc)
{
    int id = proc.world_rank;
    int world_size = proc.world_size;

    com_port com(id, world_size);

    std::vector<double> data(10, id);
    size_t data2 = id;

    std::vector<double> recv_data(10);
    size_t recv_data2;

    // send stuff in a circle
    int next_rank = (id + 1) % world_size;
    int prev_rank = (id + world_size - 1) % world_size;

    std::cout << "\n\nStarting com test" << std::endl;

    std::cout << id << ": Sending to " << next_rank << ", receiving from " << prev_rank << std::endl;

    for (int i = 0; i < world_size + 2; i++)
    { // Repeat many times to expose potential issues
        // Data should do a full circle + 1
        com_request recv_req = com.receive_begin(prev_rank, recv_data, recv_data2);
        com_request send_req = com.send_begin(next_rank, data, data2);

        com.wait(send_req, recv_req);
        std::swap(data, recv_data);
        std::swap(data2, recv_data2);
    }

    std::cout << id << ": ";
    for (auto &elem : recv_data)
        std::cout << elem << " ";
    std::cout << recv_data2 << std::endl;
}

int main(int argc, char **argv) 
{

    mpi_process proc;

    //test_com(proc);

    if (proc.world_rank == 0)
    {
        size_t size = std::stoi(argv[1]); 
        size_t dim = std::stoi(argv[2]);
        size_t k = std::stoi(argv[3]);
        test_knn(size, dim, k, 0, size);
    }
    return 0;
}

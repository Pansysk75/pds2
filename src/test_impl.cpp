#include <iostream>
#include <string>
#include <vector>

#include "detail/fileio.hpp"
#include "detail/knn_algorithms.hpp"
#include "detail/knn_utils.hpp"
#include "detail/utilities.hpp"

#include <cblas.h>
#include <omp.h>

bool compareResults(ResultPacket &p1, ResultPacket &p2)
{
    bool flag = (p1.k == p2.k) && (p1.m_packet == p2.m_packet) &&
                (p1.n_packet == p2.n_packet);

    if (flag == false)
    {
        return false;
        std::cout << "Mistake in flags" << std::endl;
    }

    // For every point in query, if knn indices are the same
    flag = true;
    for (size_t i = 0; (i < p1.m_packet) && (flag == true); i++)
    {
        auto p1_begin = p1.nidx.begin() + i * p1.k;
        auto p1_end = p1.nidx.begin() + (i + 1) * p1.k;
        auto p2_begin = p2.nidx.begin() + i * p1.k;

        if (!std::is_permutation(p1_begin, p1_end, p2_begin))
        {
            std::cout << "Mistake in indices" << std::endl;
            std::cout << "i = " << i << std::endl;
            for (size_t j = 0; j < p1.k; j++)
            {
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

void test_knn(size_t size, size_t dim, size_t k_arg)
{


    utilities::timer timer;

    size_t m = size/5;

    bool use_regular = false;
    auto [query, corpus, k] =
        use_regular ? regular_grid(size, dim, m)
                    : [size, dim, k_arg, m]() -> std::tuple<QueryPacket, CorpusPacket, size_t>
    {
        auto [q, c] = random_grid(m, size, dim);
        return std::make_tuple(q, c, k_arg);
    }();

    // Cool Impl
    timer.start();
    ResultPacket rb = knn_blas(query, corpus, k);
    timer.stop();
    auto tb = timer.get() / 1000000;

    std::cout << "Blas: " << tb << "ms " << std::endl;

    // Simple Impl
    timer.start();
    ResultPacket rs = knn_simple(query, corpus, k);
    timer.stop();
    auto ts = timer.get() / 1000000;

    std::cout << "Simple: " << ts << "ms " << std::endl;

    // split for memory
    timer.start();
    ResultPacket rbp = knn_blas_in_parts(query, corpus, k);
    timer.stop();
    auto tbp = timer.get() / 1000000;
    std::cout << "Blas in Parts: " << tbp << "ms " << std::endl;

    bool eqsb = compareResults(rs, rb);
    bool eqsbp = compareResults(rs, rbp);

    std::cout << "Completed test" << std::endl;

    std::cout << "Equality Test (simple vs blas): " << eqsb << std::endl;
    std::cout << "Equality Test (simple vs blas in parts): " << eqsbp << std::endl;
    std::cout << "Num threads omp: " << omp_get_max_threads() << std::endl; 
   // std::cout << "Num threads blas: " << openblas_get_num_threads() << std::endl;
/*

    auto factorial = [](int a) {
        int res = 1;
        while(a > 1) {
            res *= a--;
        }
        return res;
    };

    auto power = [](int a, int b) {
        int c = 1;
        for(int i = 0; i < b; i++) {
            c *= a;
        }
        return c;
    };


    std::vector<int> correct(dim+1);
    for (size_t d2 = 0; d2 <= dim; d2++) {
        correct[d2] = factorial(dim) / (factorial(d2) * factorial(dim - d2)) * power(2, d2);
    }

    for(auto c: correct) {
        std::cout << c << std::endl;
    }

    auto correctness = [dim, correct, &factorial, &power](ResultPacket &r) {
        for(size_t i = 0; i < r.m_packet; i ++) {
            std::vector<int> amts(dim, 0);
            for(size_t j = 0; j < r.k; j++) {
                int d2 = (int)(r.ndist[idx(i, j, r.k)] + 0.1);
                amts[d2]++;
            }

            for(size_t d2 = 0; d2 < dim; d2++) {
                if(amts[d2] != correct[d2]) {
                    std::cout << "Error in i = " << i << ", d2 = " << d2  << std::endl;
                    std::cout << "Correct: " << correct[d2] << " Actual: " << amts[d2] << std::endl;
                }
            }
        }
        return true;
    };
    
    bool cs = correctness(rs);
    bool c2 = correctness(r2);
    bool c3 = correctness(r3);

    std::cout << "Correctness Test (1): " << c1 << std::endl;
    std::cout << "Correctness Test Simple: " << cs << std::endl;
    std::cout << "Correctness Test (3): " << c3 << std::endl;
*/
}

int main(int argc, char **argv)
{

    if (argc != 6 && argc != 7)
    {
        std::cout << "Usage: ./tests <size> <dim> <k>"
                  << std::endl;
        return 1;
    }

    size_t size = std::stoi(argv[1]);
    size_t dim = std::stoi(argv[2]);
    size_t k = std::stoi(argv[3]);
    size_t omp_t = std::stoi(argv[4]);
    size_t blas_t = std::stoi(argv[5]);

    test_knn(size, dim, k);

    return 0;
}

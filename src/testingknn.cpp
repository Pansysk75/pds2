#include <random>
#include <chrono>
#include <tuple>
#include <omp.h>

#include "knnDist.hpp"
#include "testingknn.hpp"
#include "fileio.hpp"
#include "global_includes.hpp"

// RowMajor 3d grid up to SxSxS
// m is query points
std::tuple<QueryPacket, CorpusPacket, size_t> regual_grid(size_t s, size_t d, size_t m) {
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

    return std::make_tuple(
        QueryPacket(m, d, 0, m, std::move(X)),
        CorpusPacket(n, d, 0, n, std::move(Y)),
        k
    );
}

std::tuple<QueryPacket, CorpusPacket> random_grid(size_t m, size_t n, size_t d, size_t k) {
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

    return std::make_tuple(
        QueryPacket(m, d, 0, m, std::move(X)),
        CorpusPacket(n, d, 0, n, std::move(Y))
    );
}

std::tuple<QueryPacket, CorpusPacket> file_packets(
    const std::string& query_path, const size_t m,
    const std::string& corpus_path, const size_t n,
    const size_t d
) {
    std::vector<double> X = load_csv<double>(query_path, m);

    std::vector<double> Y = load_csv<double>(corpus_path, n);

    return std::make_tuple(
        QueryPacket(m, d, 0, m, std::move(X)),
        CorpusPacket(n, d, 0, n, std::move(Y))
    );
}

ResultPacket SyskoSimulation(
    const QueryPacket& query, const CorpusPacket& corpus, size_t k,
    const size_t num_batches_x, const size_t num_batches_y
) {

    std::vector<ResultPacket> diffProcRes;
    diffProcRes.reserve(num_batches_x);

    size_t batch_size_x = query.m_packet / num_batches_x;
    size_t batch_size_y = corpus.n_packet / num_batches_y;

    // different proccesses
    for(size_t p_id = 0; p_id < num_batches_x; p_id++){
        size_t x_start_index = p_id * batch_size_x;
        size_t x_end_index = p_id == num_batches_x - 1 ? query.x_end_index : (p_id + 1) * batch_size_x;

        QueryPacket proc_X = QueryPacket(
            batch_size_x, query.d,
            x_start_index, x_end_index,
            std::vector<double>(&query.X[idx(x_start_index, 0, query.d)], &query.X[idx(x_end_index, 0, query.d)])
        );

        ResultPacket proc_res = ResultPacket(
            0, 0, 0, 0, 0, 0, 0
        );

        // different execution batches, each begins in a different stage
        // for example stage 0 could be y[0:100]
        // stage 1 y[100:200] etc
        // they cycle through all the stages
        for(size_t stage = p_id, iters = 0; iters < num_batches_y; iters++){
            stage = (stage + 1)%num_batches_y;
            size_t y_start_index = stage * batch_size_y;
            size_t y_end_index = stage == num_batches_y - 1 ? corpus.y_end_index : (stage + 1) * batch_size_y;

            CorpusPacket proc_Y = CorpusPacket(
                batch_size_y, corpus.d,
                y_start_index, y_end_index,
                std::vector<double>(&corpus.Y[idx(y_start_index, 0, corpus.d)], &corpus.Y[idx(y_end_index, 0, corpus.d)])
            );

            ResultPacket batch_res(proc_X, proc_Y, k);
            proc_res = ResultPacket::combineKnnResultsSameX(proc_res, batch_res);
        }

        diffProcRes.push_back(std::move(proc_res));
    }

    return ResultPacket::combineCompleteQueries(diffProcRes);
}

ResultPacket runData(const QueryPacket& query, const CorpusPacket& corpus, size_t k) {
    auto start = std::chrono::high_resolution_clock::now();
    ResultPacket result(query, corpus, k);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;

    return result;
}

ResultPacket runDistrData(const QueryPacket& query, const CorpusPacket& corpus, size_t k, size_t num_batches_x, size_t num_batches_y) {
    auto start = std::chrono::high_resolution_clock::now();
    ResultPacket result = SyskoSimulation(query, corpus, k, num_batches_x, num_batches_y);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;

    return result;
}
#pragma once

#include "knn_structs.hpp"
#include <tuple>

// RowMajor d dimentions grid up to s on each side
// m is query points, picket at random from corpus
std::tuple<QueryPacket, CorpusPacket, size_t> regual_grid(size_t s, size_t d, size_t m);

std::tuple<QueryPacket, CorpusPacket> random_grid(size_t m, size_t n, size_t d, size_t k);

std::tuple<QueryPacket, CorpusPacket> file_packets(const std::string &query_path, const size_t query_start_idx,  const size_t query_end_idx,const std::string &corpus_path, const size_t corpus_start_idx,  const size_t corpus_end_idx, const size_t d_upper_limit);

std::tuple<QueryPacket, CorpusPacket> file_packets(const std::string &file_path, const size_t start_idx,  const size_t end_idx, const size_t d_upper_limit);

ResultPacket SyskoSimulation(const QueryPacket &query, const CorpusPacket &corpus, size_t k, const size_t num_batches_x, const size_t num_batches_y);

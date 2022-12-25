# pragma once

#include <vector>
#include <tuple>

#include "knnDist.hpp"
#include "global_vars.hpp"

// RowMajor 3d grid up to SxSxS
// m is query points
std::tuple<QueryPacket, CorpusPacket, size_t> regual_grid(size_t s, size_t d, size_t m);

std::tuple<QueryPacket, CorpusPacket> random_grid(size_t m, size_t n, size_t d, size_t k);

ResultPacket runData(const QueryPacket& query, const CorpusPacket& corpus, size_t k);
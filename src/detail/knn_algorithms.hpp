#include "knn_structs.hpp"

ResultPacket knn_blas(const QueryPacket &query,
                      const CorpusPacket &corpus, size_t k_arg);

ResultPacket knn_simple(const QueryPacket &query, const CorpusPacket &corpus, size_t k_arg);

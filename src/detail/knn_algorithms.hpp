#include "knn_structs.hpp"

ResultPacket knn_blas(const QueryPacket &query,
                      const CorpusPacket &corpus, size_t k_arg);

ResultPacket knn_blas_in_parts(const QueryPacket &query, const CorpusPacket &corpus, const size_t k_arg, const size_t parts);

ResultPacket knn_simple(const QueryPacket &query, const CorpusPacket &corpus, size_t k_arg);

ResultPacket knn_dynamic(const QueryPacket &query, const CorpusPacket &corpus, size_t k_arg);

#include <iostream>

#include "knn_structs.hpp"

CorpusPacket::CorpusPacket(size_t n_packet, size_t d, size_t y_start_index,
                           size_t y_end_index)
    : n_packet(n_packet), d(d), y_start_index(y_start_index),
      y_end_index(y_end_index) {
  Y = std::vector<double>(n_packet * d);
}

CorpusPacket::CorpusPacket(size_t n_packet, size_t d, size_t y_start_index,
                           size_t y_end_index, std::vector<double> &&Y)
    : n_packet(n_packet), d(d), y_start_index(y_start_index),
      y_end_index(y_end_index), Y(std::move(Y)) {}

QueryPacket::QueryPacket(size_t m_packet, size_t d, size_t x_start_index,
                         size_t x_end_index)
    : m_packet(m_packet), d(d), x_start_index(x_start_index),
      x_end_index(x_end_index) {
  X = std::vector<double>(m_packet * d);
}

QueryPacket::QueryPacket(size_t m_packet, size_t d, size_t x_start_index,
                         size_t x_end_index, std::vector<double> &&X)
    : m_packet(m_packet), d(d), x_start_index(x_start_index),
      x_end_index(x_end_index), X(std::move(X)) {}

// Inconsistent with the other packets
ResultPacket::ResultPacket(size_t vec_size) {
  nidx.resize(vec_size);
  ndist.resize(vec_size);
  m_packet = 0;
  n_packet = 0;
  k = 0;
  x_start_index = 0;
  x_end_index = 0;
  y_start_index = 0;
  y_end_index = 0;
}

ResultPacket::ResultPacket() : ResultPacket(0) {}

// this is the constructor for the result packet, without it being solved.
// It needs to be filled manually
ResultPacket::ResultPacket(size_t m_packet, size_t n_packet, size_t k_arg,
                           size_t x_start_index, size_t x_end_index,
                           size_t y_start_index, size_t y_end_index)
    : m_packet(m_packet), n_packet(n_packet), k(std::min(k_arg, n_packet)),
      x_start_index(x_start_index), x_end_index(x_end_index),
      y_start_index(y_start_index), y_end_index(y_end_index) {
  nidx = std::vector<size_t>(m_packet * this->k);
  ndist = std::vector<double>(m_packet * this->k);
}
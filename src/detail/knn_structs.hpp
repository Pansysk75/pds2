#pragma once

#include <iostream>
#include <vector>

struct CorpusPacket {
  size_t n_packet;
  size_t d;

  size_t y_start_index;
  size_t y_end_index;

  std::vector<double> Y;

  // empty constructor
  CorpusPacket() {}

  CorpusPacket(size_t n_packet, size_t d, size_t y_start_index,
               size_t y_end_index);

  // moving Y into the packet
  CorpusPacket(size_t n_packet, size_t d, size_t y_start_index,
               size_t y_end_index, std::vector<double> &&Y);
};

// make printable

inline std::ostream &operator<<(std::ostream &os, const CorpusPacket &c) {
  os << c.y_start_index << "->" << c.y_end_index << " | n:" << c.n_packet
     << " d:" << c.d;
  return os;
}

// make sendable

struct QueryPacket {
  size_t m_packet;
  size_t d;

  size_t x_start_index;
  size_t x_end_index;

  std::vector<double> X;

  // empty constructor
  QueryPacket() {}

  QueryPacket(size_t m_packet, size_t d, size_t x_start_index,
              size_t x_end_index);

  // moving X into the packet
  QueryPacket(size_t m_packet, size_t d, size_t x_start_index,
              size_t x_end_index, std::vector<double> &&X);
};

// make printable

inline std::ostream &operator<<(std::ostream &os, const QueryPacket &c) {
  os << c.x_start_index << "->" << c.x_end_index << " | n:" << c.m_packet
     << " d:" << c.d;
  return os;
}

struct ResultPacket {

  size_t m_packet;
  size_t n_packet;
  size_t k;

  size_t x_start_index;
  size_t x_end_index;

  // if y_end_index < y_start_index, then it wraps around
  // for example if y_start_index = 0 and y_end_index = 1000, then it is the
  // first 1000 points if y_start_index = 200 and y_end_index = 100 and
  // n_packet = 500 then it is the points 200:600 and 0:100
  size_t y_start_index;
  size_t y_end_index;

  // in global index of y
  std::vector<size_t> nidx;
  std::vector<double> ndist;

  // empty constructor
  ResultPacket();

  ResultPacket(size_t vec_size);

  // this is the constructor for the result packet, without it being solved.
  // It needs to be filled manually
  ResultPacket(size_t m_packet, size_t n_packet, size_t k_arg,
               size_t x_start_index, size_t x_end_index, size_t y_start_index,
               size_t y_end_index);

  // this is the solver, it takes a query and a corpus and returns a result
  ResultPacket(const QueryPacket &query, const CorpusPacket &corpus, size_t k);
};

// make printable
inline std::ostream &operator<<(std::ostream &os, const ResultPacket &c) {
  os << "X:" << c.x_start_index << "->" << c.x_end_index
     << " | Y:" << c.y_start_index << "->" << c.y_end_index;
  return os;
}

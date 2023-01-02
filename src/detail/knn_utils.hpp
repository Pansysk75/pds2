#include "knn_structs.hpp"
#include <tuple>

size_t idx(size_t i, size_t j, size_t ld);

// they need to be distances of
// SAME query points
// DIFFERENT corpus points
std::tuple<bool, size_t, size_t> combinableSameX(const ResultPacket &back,
                                                 const ResultPacket &front);

// they need to be distances of
// DIFFERENT query points
// SAME corpus points
std::tuple<bool, size_t, size_t> combinableSameY(const ResultPacket &back,
                                                 const ResultPacket &front);

// they need to be combinableSameX have back.x_end_index == front.x_start_index
// for example we combine the k nearest neighbors of x[0:100] in both results
// but the first is the k nearest neighbors from y[0:100] and the second is the
// k nearest neighbors from y[100:200]
ResultPacket combineKnnResultsSameX(const ResultPacket &back,
                                    const ResultPacket &front);

// they need to be combinableSameY have back.y_end_index == front.y_start_index
// for example we combine the k nearest neighbors from y[0:100] in both results
// the first is the k nearest neighbors of x[0:100] and the second is the k
// nearest neighbors of x[100:200]
ResultPacket combineKnnResultsSameY(const ResultPacket &back,
                                    const ResultPacket &front);

// they all share the same Y (which is the whole Y) and collectivly cover the
// whole X
ResultPacket combineCompleteQueries(std::vector<ResultPacket> &results);
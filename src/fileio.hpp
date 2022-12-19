#pragma once

#include <iostream>
#include <vector>

template <typename T>
std::vector<T> load_csv(const std::string& filename, const size_t upper_limit);
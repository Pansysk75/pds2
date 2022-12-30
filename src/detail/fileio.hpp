#pragma once

#include <iostream>
#include <vector>
#include <tuple>

template <typename T>
std::tuple<std::vector<T>, size_t, size_t> load_csv(const std::string& filename, const size_t line_upper_limit, const size_t el_upper_limit, const bool skip_first_line, const bool skip_first_field);

// ! Imitates importing data !
std::vector<double> import_data(int idx_start, int idx_end, int dim);
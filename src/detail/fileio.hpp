#pragma once

#include <iostream>
#include <tuple>
#include <vector>

template <typename T>
std::tuple<std::vector<T>, size_t, size_t>
load_csv(const std::string &filename, const size_t line_begin,
         const size_t line_end, const size_t el_upper_limit,
         const bool skip_first_field);

template <typename T>
std::tuple<std::vector<T>, size_t, size_t, std::vector<std::string>>
load_csv_with_labels(const std::string &filename, const size_t line_begin,
                     const size_t line_end, const size_t el_upper_limit);

void vectorToCSV(std::string filename, const std::vector<double>& vec, const size_t lines, const size_t dim, bool pad);

// ! Imitates importing data !
std::vector<double> import_data(int idx_start, int idx_end, int dim);
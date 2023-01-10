#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#define UNASSIGNED -1
#define NO_COLOR -1

#include "fileio.hpp"

template <typename T>
std::tuple<std::vector<T>, size_t, size_t>
load_csv(const std::string& filename, const size_t line_begin,
         const size_t line_end, const size_t el_upper_limit,
         const bool skip_first_field)
{
    std::vector<T> data;

    // if the file exists
    if (!std::filesystem::exists(filename))
    {
        throw std::runtime_error("File '" + filename + "' does not exist.");
    }

    size_t lines_got = 0;
    std::ifstream file(filename);
    if (file.is_open())
    {
        // skip first "line_begin" lines
        for (unsigned int _ = 0; _ < line_begin + 1; _++)
        {
            file.ignore(std::numeric_limits<std::streamsize>::max(),
                        file.widen('\n'));
        }
        const size_t n_lines = line_end - line_begin;
        std::string line;
        while (std::getline(file, line) && lines_got < n_lines)
        {
            if (skip_first_field)
                line = line.substr(line.find_first_of(",") + 1);

            std::stringstream lineStream(line);
            std::string value;
            size_t elements_got = 0;
            while (std::getline(lineStream, value, ',') &&
                   elements_got < el_upper_limit)
            {
                T convertedValue;
                std::stringstream(value) >> convertedValue;
                data.push_back(convertedValue);
                elements_got++;
            }
            lines_got++;
        }
    }

    return std::make_tuple(std::move(data), lines_got, data.size() / lines_got);
}

// instanciate the template for common numeric types

template std::tuple<std::vector<int>, size_t, size_t>
load_csv<int>(const std::string &filename, const size_t line_begin,
              const size_t line_end, const size_t el_upper_limit,
              const bool skip_first_field);

template std::tuple<std::vector<size_t>, size_t, size_t>
load_csv<size_t>(const std::string &filename, const size_t line_begin,
                 const size_t line_end, const size_t el_upper_limit,
                 const bool skip_first_field);

template std::tuple<std::vector<float>, size_t, size_t>
load_csv<float>(const std::string &filename, const size_t line_begin,
                const size_t line_end, const size_t el_upper_limit,
                const bool skip_first_field);

template std::tuple<std::vector<double>, size_t, size_t>
load_csv<double>(const std::string &filename, const size_t line_begin,
                 const size_t line_end, const size_t el_upper_limit,
                 const bool skip_first_field);

template <typename T>
std::tuple<std::vector<T>, size_t, size_t, std::vector<std::string>>
load_csv_with_labels(const std::string &filename, const size_t line_begin,
                     const size_t line_end, const size_t el_upper_limit) {

    // check if the file exists
    if (!std::filesystem::exists(filename))
    {
        throw std::runtime_error("File '" + filename + "' does not exist.");
    }

    std::vector<T> data;
    data.reserve((line_end - line_begin) * el_upper_limit);
    std::vector<std::string> labels;
    labels.reserve(line_end - line_begin);

    size_t lines_got = 0;
    std::ifstream file(filename);
    if (file.is_open())
    {
        // skip first "line_begin" lines
        for (unsigned int _ = 0; _ < line_begin + 1; _++)
        {
            file.ignore(std::numeric_limits<std::streamsize>::max(),
                        file.widen('\n'));
        }
        const size_t n_lines = line_end - line_begin;
        std::string line;
        while (std::getline(file, line) && lines_got < n_lines)
        {
            // get the label
            size_t pos = line.find_first_of(",");
            labels.push_back(line.substr(0, pos));

            // skip it for the rest of the line
            line = line.substr(pos + 1);

            std::stringstream lineStream(line);
            std::string value;
            size_t elements_got = 0;
            while (std::getline(lineStream, value, ',') &&
                   elements_got < el_upper_limit)
            {
                T convertedValue;
                std::stringstream(value) >> convertedValue;
                data.push_back(convertedValue);
                elements_got++;
            }
            lines_got++;
        }
    }

    return std::make_tuple(std::move(data), lines_got, data.size() / lines_got, std::move(labels));
}

template std::tuple<std::vector<int>, size_t, size_t, std::vector<std::string>>
load_csv_with_labels<int>(const std::string &filename, const size_t line_begin,
                     const size_t line_end, const size_t el_upper_limit);

template std::tuple<std::vector<size_t>, size_t, size_t, std::vector<std::string>>
load_csv_with_labels<size_t>(const std::string &filename, const size_t line_begin,
                     const size_t line_end, const size_t el_upper_limit);

template std::tuple<std::vector<float>, size_t, size_t, std::vector<std::string>>
load_csv_with_labels<float>(const std::string &filename, const size_t line_begin,
                     const size_t line_end, const size_t el_upper_limit);

template std::tuple<std::vector<double>, size_t, size_t, std::vector<std::string>>
load_csv_with_labels<double>(const std::string &filename, const size_t line_begin,
                     const size_t line_end, const size_t el_upper_limit);

void vectorToCSV(std::string filename, const std::vector<double>& vec, const size_t lines, const size_t dim, bool pad) {
    std::ofstream file(filename);
    if (pad) {
        for (size_t j = 0; j < dim+1; j++) {
            file << 0;
            if (j < dim) {
                file << ",";
            }
        }
        file << "\n";
    }
    for (size_t i = 0; i < lines; i++) {
        if (pad) {
            file << "0,";
        }
        for (size_t j = 0; j < dim; j++) {
            file << vec[i*dim + j];
            if (j < dim - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
    file.close();
}

std::vector<double> import_data(int idx_start, int idx_end, int dim){
    // Imitates importing data
    int size = idx_end - idx_start;
    std::vector<double> vec(size*dim);
    std::iota(vec.begin(), vec.end(), idx_start*dim);
    return vec;
}
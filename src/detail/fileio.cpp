#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <tuple>
#include <sstream>
#include <filesystem>
#include <algorithm>

#define UNASSIGNED -1
#define NO_COLOR -1

#include "fileio.hpp"
#include "global_includes.hpp"

template <typename T>
std::tuple<std::vector<T>, size_t, size_t> load_csv(
    const std::string& filename, 
    const size_t line_upper_limit, const size_t el_upper_limit, 
    const bool skip_first_line, const bool skip_first_field
) {
    std::vector<T> data;

    // if the file exists
    if(!std::filesystem::exists(filename)) {
        std::cout << "File does not exist" << std::endl;
        return std::make_tuple(std::move(data), 0, 0);
    }

    size_t lines_got = 0;
    std::ifstream file(filename);
    if (file.is_open())
    {
        // skip the first line
        if(skip_first_line)
            file.ignore(std::numeric_limits<std::streamsize>::max(), file.widen('\n'));
        std::string line;
        while (std::getline(file, line) && lines_got++ < line_upper_limit)
        {
            if(skip_first_field)
                line = line.substr(line.find_first_of(",") + 1);

            std::stringstream lineStream(line);
            std::string value;
            size_t elements_got = 0;
            while (std::getline(lineStream, value, ',') && elements_got++ < el_upper_limit)
            {
                T convertedValue;
                std::stringstream(value) >> convertedValue;
                data.push_back(convertedValue);
            }
        }
    }

    return std::make_tuple(std::move(data), lines_got, data.size() / lines_got);
}


// instanciate the template for common numeric types

template std::tuple<std::vector<int>, size_t, size_t> load_csv<int>(const std::string& filename, const size_t line_upper_limit, const size_t el_upper_limit, const bool skip_first_line, const bool skip_first_field);
template std::tuple<std::vector<size_t>, size_t, size_t> load_csv<size_t>(const std::string& filename, const size_t line_upper_limit, const size_t el_upper_limit, const bool skip_first_line, const bool skip_first_field);
template std::tuple<std::vector<float>, size_t, size_t> load_csv<float>(const std::string& filename, const size_t line_upper_limit, const size_t el_upper_limit, const bool skip_first_line, const bool skip_first_field);
template std::tuple<std::vector<double>, size_t, size_t> load_csv<double>(const std::string& filename, const size_t line_upper_limit, const size_t el_upper_limit, const bool skip_first_line, const bool skip_first_field);
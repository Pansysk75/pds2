#include <iostream>
#include <string>
#include <fstream>
#include <array>
#include <sstream>
#include <filesystem>
#include <algorithm>

#define UNASSIGNED -1
#define NO_COLOR -1

#include "fileio.hpp"
#include "global_includes.hpp"

template <typename T>
std::vector<T> load_csv(const std::string& filename, const size_t upper_limit)
{
    std::vector<T> data;

    // if the file exists
    if(!std::filesystem::exists(filename)) {
        std::cout << "File does not exist" << std::endl;
        return data;
    }

    size_t lines_got = 0;
    std::ifstream file(filename);
    if (file.is_open())
    {
        // skip the first line
        file.ignore(std::numeric_limits<std::streamsize>::max(), file.widen('\n'));
        std::string line;
        while (std::getline(file, line) && lines_got++ < upper_limit)
        {
            std::stringstream lineStream(line);
            std::string value;
            while (std::getline(lineStream, value, ','))
            {
                T convertedValue;
                std::stringstream(value) >> convertedValue;
                data.push_back(convertedValue);
            }
        }
    }

    return data;
}


// instanciate the template for common numeric types

template std::vector<int> load_csv<int>(const std::string& filename, const size_t upper_limit);
template std::vector<size_t> load_csv<size_t>(const std::string& filename, const size_t upper_limit);
template std::vector<float> load_csv<float>(const std::string& filename, const size_t upper_limit);
template std::vector<double> load_csv<double>(const std::string& filename, const size_t upper_limit);

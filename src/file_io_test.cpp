#include "detail/testingknn.hpp"


int main(){
    //hard coded for now
    std::string filename = "datasets/mnist_test.csv";
    size_t n_total = 10000; // # total points
    size_t d = 28*28;         // # dimensions
    size_t k = 3;          // # nearest neighbours
    auto [query, corpus] = file_packets(filename, 0, 10, 3);
    return 0;
}

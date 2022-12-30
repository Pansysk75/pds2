#include <iostream>
#include <string>
#include <vector>

#include "detail/knnDist.hpp"
#include "detail/utilities.hpp"
#include "detail/fileio.hpp"
#include "detail/testingknn.hpp"

bool compareResults(ResultPacket& p1, ResultPacket& p2){
    bool metadata_check = 
        (p1.k == p2.k) &&
        (p1.m_packet == p2.m_packet) &&
        (p1.n_packet == p2.n_packet);

    // Check if results contain the same indices
    bool idx_check = std::is_permutation(p1.nidx.begin(), p1.nidx.end(), p2.nidx.begin());

    return metadata_check && idx_check;
}


void test(size_t size, size_t dim, size_t k, size_t idx_start, size_t idx_end){

    utilities::timer timer;

    QueryPacket query(size, dim, idx_start, idx_end);
    query.X = import_data(idx_start, idx_end, dim);

    CorpusPacket corpus(size, dim, idx_start, idx_end);
    corpus.Y = import_data(idx_start, idx_end, dim);

    // Simple Impl
    timer.start();
    ResultPacket r1 = simpleKnn(query, corpus, k);
    timer.stop();
    auto t1 = timer.get()/1000000;

    // Cool Impl
    timer.start();
    ResultPacket r2(query, corpus, k);
    timer.stop();
    auto t2 = timer.get()/1000000;

    bool comp_result = compareResults(r1, r2);

    std::cout << "Completed test\n t1 = " << t1 << " ms\n t2 = " << t2 << " ms\n Equality Test = " << comp_result << std::endl;
    
}

int main(){
    

    test(100, 2, 5, 0, 100);

    return 0;
}
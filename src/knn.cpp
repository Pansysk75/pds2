#include <vector>
#include <algorithm>
#include <iostream>

// NOTE: This is ONLY a SEQUENTIAL kNN version, it's not yet
// compatible to the other utilities in this repo
// Work in progress :)


struct knnresult{
    std::vector<int> nidx;  // Indices (0-based) of nearest neighbors [m-by-k]
    std::vector<double> ndist;  // Distance of nearest neighbors [m-by-k]
    int m;       //!< Number of query points
    int k;       //!< Number of nearest neighbors

    knnresult(int m, int k)
    :m(m), k(k){
        nidx.reserve(m*k);
        ndist.reserve(m*k);
    }
};



struct index_distance_pair{
    // Pair index & distance in a class, so we can easily
    // store them in a data structure and sort them based on distance.
    double distance;
    int index;
    bool operator<(index_distance_pair const& rh) const{
        return distance < rh.distance;
    }
};

template <typename T>
struct bounded_max_heap{
    // It's a max heap, meaning that the maximum element
    // can be accessed in O(1).
    // It's bounded, meaning it can only store a limited
    // number of elements
    std::vector<T> data;
    unsigned int max_size;

    bounded_max_heap<T>(unsigned int max_size)
    :max_size(max_size)
    {
        data.reserve(max_size);
    }

    T& top(){
        return data[0];
    }

    void clear(){
        data.clear();
    }

    void insert(T&& elem){
        // Insert an element.
        // If heap is full, replaces the top element.
        if(data.size() < max_size){
            data.push_back(elem);
            std::make_heap(data.begin(), data.end());       
        }
        else if (elem < data[0]){
            data[0] = std::move(elem);
            std::make_heap(data.begin(), data.end());       
        }
    }
};

void copy_result(int i_query, knnresult& knn_result, std::vector<index_distance_pair>& data){
    // Copies data to knn_result at position i_query
    int k = knn_result.k;
    int m = knn_result.m;
    for(int i=0; i<k; i++){
        knn_result.nidx[i_query*k + i] = data[i].index;
        knn_result.ndist[i_query*k + i] = data[i].distance;
    }
}


double calc_distance(int d, double* element1, double* element2){
    //Calculates euclidean distance between element1 and element2
    double sum = 0;
    for(int i=0; i<d; i++){
        sum += (*(element1 + i) - *(element2 + i)) * (*(element1 + i) - *(element2 + i));
    }
    return sum;
}



knnresult kNN(double * X, double * Y, int n, int m, int d, int k){

    knnresult knn_result(m, k);
        
    // The min_heap keeps track of the k elements
    // that have had the smallest distance so far.
    bounded_max_heap<index_distance_pair> heap(k);

    //for every point in Query, find k closest Corpus points
    for(int i_query=0; i_query<m; i_query++){

        heap.clear();

        for(int i_corpus=0; i_corpus<n; i_corpus++){

            // Calculate distance and add to heap if needed!
            double dist = calc_distance(d, X + i_query*d, Y + i_corpus*d);

            heap.insert({dist, i_corpus});
        }

        // Now the heap should have the kNN. Copy it to the results.
        copy_result(i_query, knn_result, heap.data);

    }

    return knn_result;
}

int main(){
    int d = 2; // number of dimensions
    int m = 4; // number of points in X
    int n = 4; // number of points in Y
    int k = 2; // number of "nearest neighbours to find"

    // Make some example points and test kNN algorithm

    std::vector<double> X{  2, 2, 
                            1, 1,
                            6, 2,
                            8, 1};

    std::vector<double> Y{  1, 2, 
                            5, 2,
                            10, 20,
                            0, 0};


    auto r = kNN(
        X.data(), Y.data(), n, m, d, k
        );


    // Print results
    std::cout << "index" << "\t" << "dist" << std::endl;

    for(int i=0; i<k*m; i++){
        if (i%k == 0) std::cout << "\n";
        std::cout << r.nidx[i] << "\t" << r.ndist[i] << std::endl;
    }


}




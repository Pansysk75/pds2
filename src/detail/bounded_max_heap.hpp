#pragma once

#include <vector>
#include <algorithm>

template <typename T>
struct bounded_max_heap
{
    // It's a max heap, meaning that the maximum element
    // can be accessed in O(1).
    // It's bounded, meaning it can only store a limited
    // number of elements
    std::vector<T> data;
    unsigned int max_size;

    bounded_max_heap(unsigned int max_size)
        : max_size(max_size)
    {
        data.reserve(max_size);
    }

    T &top()
    {
        return data[0];
    }

    void clear()
    {
        data.clear();
    }

    void insert(T &&elem)
    {
        // Insert an element.
        // If heap is full, replaces the top element.
        if (data.size() < max_size)
        {
            data.push_back(elem);
            std::make_heap(data.begin(), data.end());
        }
        else if (elem < data[0])
        {
            data[0] = std::move(elem);
            std::make_heap(data.begin(), data.end());
        }
    }
};
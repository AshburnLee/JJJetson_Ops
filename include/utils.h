#ifndef UTILS_H_
#define UTILS_H_

#pragma once

#include <iostream>
#include <stdexcept>
#include <random>

enum data_type {
    DT_F32  = 0,  // float32
    DT_BF16 = 1,  // bfloat16
    DT_F16  = 2,  // half
    DT_I32  = 3,  // int32
};

template<typename T>
void InitInputData(T* addr, int numl) {
    if (numl <= 0) {
        throw std::invalid_argument("numl must be positive");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(1.0, 10.0);

    for (int i = 0; i < numl; ++i) {
        addr[i] = dis(gen);
    }
    return;
}

template <typename T>
void ShowSamples(T* ptr) {
    std::cout << ptr[0] << " ";
    std::cout << ptr[1] << " ";
    std::cout << ptr[2] << " ";
    std::cout << ptr[3] << " ";
    std::cout << ptr[4] << " ";
    std::cout << ptr[5] << " ";
    std::cout << ptr[6] << "\n";
}

#endif 

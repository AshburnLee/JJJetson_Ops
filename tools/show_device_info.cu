// 仅用于在 host 侧打印当前 GPU 信息
#include "device_hw_info.cuh"

#include <cstdlib>

int main(int argc, char **argv) {
    int device_id = 0;
    if (argc > 1) {
        device_id = std::atoi(argv[1]);
    }

    cudaError_t st = cudaSetDevice(device_id);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", device_id, cudaGetErrorString(st));
        return 1;
    }

    const DeviceHwInfo h = query_device_hw_info(device_id);
    fprint_device_hw_info(stdout, h);
    return 0;
}

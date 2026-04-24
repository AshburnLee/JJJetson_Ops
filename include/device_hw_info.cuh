#ifndef DEVICE_HW_INFO_CUH_
#define DEVICE_HW_INFO_CUH_

// 理论值含义：
// - FP32：按"每 SM 的 FP32 CUDA core 数 × SM 数 × 2（FMA）× 当前报告的 SM 时钟"估算；Tensor Core、
//   混合指令与真实可持续 FLOPS 不在此公式内。
// - DRAM：按"2 × memClock × (busWidth/8)"的常见 DDR 有效字节率估算；与 ncu 的实测可能有偏差。

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

inline __host__ int device_fp32_cuda_cores_per_sm(int cc_major, int cc_minor) {
    const int cc = cc_major * 10 + cc_minor;
    switch (cc) {
        case 30:
        case 32:
        case 35:
        case 37:
            return 192;
        case 50:
        case 52:
        case 53:
            return 128;
        case 60:
            return 64;
        case 61:
        case 62:
            return 128;
        case 70:
            return 64;
        case 72:
            return 64;
        case 75:
            return 64;
        case 80:
            return 64;
        case 86:
        case 87:
        case 89:
            return 128;
        case 90:
            return 128;
        case 100:
        case 101:
        case 120:
            return 128;
        default:
            return 0;
    }
}

// Jetson / Ampere-系列：每 SM 4 个 warp 调度器，每调度器常驻 warp 数上限 12
// 故 max warps/SM = 4×12 = 48，即 max threads/SM = 48×32 = 1536。这是架构常数，不是 cudaDeviceProp
inline __host__ int device_warp_schedulers_per_sm(int cc_major, int cc_minor) {
    const int cc = cc_major * 10 + cc_minor;
    switch (cc) {
        case 75:
        case 80:
        case 86:
        case 87:
        case 89:
        case 90:
        case 100:
        case 101:
        case 120:
            return 4;
        default:
            return 0;
    }
}

// 每 warp scheduler 上的 warp 数上限（Turing=8，Ampere+ 为 12）
inline __host__ int device_max_resident_warps_per_warp_scheduler(int cc_major, int cc_minor) {
    const int cc = cc_major * 10 + cc_minor;
    switch (cc) {
        case 75:
            return 8;
        case 80:
        case 86:
        case 87:
        case 89:
        case 90:
        case 100:
        case 101:
        case 120:
            return 12;
        default:
            return 0;
    }
}

// 每 SM 每周期 可发出的指令条数理论上界
inline __host__ int device_max_instructions_issued_per_cycle_per_sm(int cc_major, int cc_minor) {
    const int sched = device_warp_schedulers_per_sm(cc_major, cc_minor);
    return sched > 0 ? sched : 0;
}

struct DeviceHwInfo {
    char name[256];
    int device_id{-1};
    int cc_major{0};
    int cc_minor{0};
    int sm_count{0};
    int sm_clock_khz{0};
    int mem_clock_khz{0};
    int mem_bus_width_bits{0};
    int fp32_cores_per_sm{0};
    float peak_fp32_tflops_theoretical{0.f};
    float peak_dram_gbps_theoretical{0.f};

    // 容量 / 并行度（主要来自 cudaDeviceProp；L1 独立容量多数架构下驱动不单独上报）
    size_t total_global_mem_bytes{0};
    size_t total_const_mem_bytes{0};
    int l2_cache_bytes{0};
    int global_l1_cache_supported{0};
    int local_l1_cache_supported{0};
    size_t shared_mem_per_block_bytes{0};
    size_t shared_mem_per_multiprocessor_bytes{0};
    int regs_per_block{0};
    int regs_per_multiprocessor{0};
    int warp_size{0};
    int max_threads_per_block{0};
    int max_threads_per_multiprocessor{0};

    int warp_schedulers_per_sm{0};
    int max_resident_warps_per_warp_scheduler{0};
    int max_warps_per_multiprocessor_from_prop{0};
    // 4 调度器×每周期 1 条/调度器 的大概上界。真实 IPC 需 profile
    int max_instructions_issued_per_cycle_per_sm_theoretical{0};
};

// 可由 host 填入后按值拷贝到 kernel 使用的轻量结构
struct DeviceRooflineParams {
    float peak_fp32_tflops{0.f};
    float peak_dram_gbps{0.f};
    int sm_count{0};
    int cc_major{0};
    int cc_minor{0};
};

inline __host__ DeviceRooflineParams to_device_roofline_params(const DeviceHwInfo& h) {
    return DeviceRooflineParams{h.peak_fp32_tflops_theoretical,
                                h.peak_dram_gbps_theoretical,
                                h.sm_count,
                                h.cc_major,
                                h.cc_minor};
}

inline __host__ DeviceHwInfo query_device_hw_info(int device_id = 0) {
    DeviceHwInfo h{};
    h.device_id = device_id;

    cudaDeviceProp prop{};
    const cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        std::snprintf(h.name, sizeof(h.name), "(cudaGetDeviceProperties: %s)", cudaGetErrorString(err));
        return h;
    }

    std::strncpy(h.name, prop.name, sizeof(h.name) - 1);
    h.name[sizeof(h.name) - 1] = '\0';

    h.cc_major = prop.major;
    h.cc_minor = prop.minor;
    h.sm_count = prop.multiProcessorCount;
    h.sm_clock_khz = prop.clockRate;
    h.mem_clock_khz = prop.memoryClockRate;
    h.mem_bus_width_bits = prop.memoryBusWidth;

    h.total_global_mem_bytes = prop.totalGlobalMem;
    h.total_const_mem_bytes = prop.totalConstMem;
#if CUDART_VERSION >= 11000
    h.l2_cache_bytes = prop.l2CacheSize;
#else
    h.l2_cache_bytes = 0;
#endif
    h.global_l1_cache_supported = prop.globalL1CacheSupported;
    h.local_l1_cache_supported = prop.localL1CacheSupported;
    h.shared_mem_per_block_bytes = prop.sharedMemPerBlock;
    h.shared_mem_per_multiprocessor_bytes = prop.sharedMemPerMultiprocessor;
    h.regs_per_block = prop.regsPerBlock;
    h.regs_per_multiprocessor = prop.regsPerMultiprocessor;
    h.warp_size = prop.warpSize;
    h.max_threads_per_block = prop.maxThreadsPerBlock;
    h.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;

    {
        const int ws = h.warp_size > 0 ? h.warp_size : 32;
        h.max_warps_per_multiprocessor_from_prop = h.max_threads_per_multiprocessor / ws;
    }
    h.warp_schedulers_per_sm = device_warp_schedulers_per_sm(h.cc_major, h.cc_minor);
    h.max_resident_warps_per_warp_scheduler = device_max_resident_warps_per_warp_scheduler(
        h.cc_major, h.cc_minor);
    h.max_instructions_issued_per_cycle_per_sm_theoretical =
        device_max_instructions_issued_per_cycle_per_sm(h.cc_major, h.cc_minor);

    h.fp32_cores_per_sm = device_fp32_cuda_cores_per_sm(h.cc_major, h.cc_minor);

    // 通用峰值算力：“并行度 × 每周期运算数 × 频率”，Jetson 上 512 x 2 x 频率
    // 单位 TFLOP/s，每秒万亿（1e12）次浮点数运算
    if (h.fp32_cores_per_sm > 0 && h.sm_clock_khz > 0 && h.sm_count > 0) {
        const double sm_hz = static_cast<double>(h.sm_clock_khz) * 1000.0;
        h.peak_fp32_tflops_theoretical = static_cast<float>(
            (static_cast<double>(h.sm_count) * static_cast<double>(h.fp32_cores_per_sm) * 2.0 * sm_hz) / 1e12);
    }

    // DRAM 峰值：字节/s = mem_hz × (busWidth_bits/8) × 2。×2 = DDR 每时钟周期在上升/下降沿各传一次；
    // busWidth/8 = 并行总线一次并行传输的字节数。mem_clock_khz 来自 cudaDeviceProp::memoryClockRate。
    // 理论带宽（字节数/s）：2(double data rate) × memClock × (busWidth/8)
    if (h.mem_clock_khz > 0 && h.mem_bus_width_bits > 0) {
        const double byte_per_sec = static_cast<double>(h.mem_clock_khz) * 1000.0 *
                                    (static_cast<double>(h.mem_bus_width_bits) / 8.0) * 2.0;
        h.peak_dram_gbps_theoretical = static_cast<float>(byte_per_sec / 1e9);
    }

    return h;
}

inline __host__ void fprint_device_hw_info(FILE* out, const DeviceHwInfo& h) {
    if (!out) {
        return;
    }

    std::fprintf(out, "[DeviceHwInfo] device_id: %d\n", h.device_id);
    std::fprintf(out, "[DeviceHwInfo] name: %s\n", h.name);
    std::fprintf(out, "[DeviceHwInfo] compute_capability: %d.%d\n", h.cc_major, h.cc_minor);
    std::fprintf(out, "[DeviceHwInfo] sm_count: %d\n", h.sm_count);
    std::fprintf(out, "[DeviceHwInfo] sm_clock_ghz: %.6f\n", static_cast<double>(h.sm_clock_khz) / 1e6);
    std::fprintf(out, "[DeviceHwInfo] mem_clock_ghz: %.6f\n", static_cast<double>(h.mem_clock_khz) / 1e6);
    std::fprintf(out, "[DeviceHwInfo] mem_bus_width_bits: %d\n", h.mem_bus_width_bits);
    std::fprintf(out, "[DeviceHwInfo] total_global_mem_gbytes: %.6f\n",
                 static_cast<double>(h.total_global_mem_bytes) / 1e9);
    // 千字节：按 1024 字节 = 1 KiB（与 CUDA 内存规格习惯一致）
    std::fprintf(out, "[DeviceHwInfo] total_const_mem_kibytes: %.6f\n",
                 static_cast<double>(h.total_const_mem_bytes) / 1024.0);
    std::fprintf(out, "[DeviceHwInfo] l2_cache_bytes: %d\n", h.l2_cache_bytes);
    std::fprintf(out, "[DeviceHwInfo] global_l1_cache_supported: %d\n", h.global_l1_cache_supported);
    std::fprintf(out, "[DeviceHwInfo] local_l1_cache_supported: %d\n", h.local_l1_cache_supported);
    std::fprintf(out, "[DeviceHwInfo] l1_data_cache_size_per_sm_bytes: N/A\n");
    std::fprintf(out,
                 "[DeviceHwInfo] l1_note: driver does not report L1-only size; L1 may share a unified SM "
                 "partition with shared memory (see shared_mem_per_multiprocessor_kibytes).\n");
    std::fprintf(out, "[DeviceHwInfo] shared_mem_per_block_kibytes: %.6f\n",
                 static_cast<double>(h.shared_mem_per_block_bytes) / 1024.0);
    std::fprintf(out, "[DeviceHwInfo] shared_mem_per_multiprocessor_kibytes: %.6f\n",
                 static_cast<double>(h.shared_mem_per_multiprocessor_bytes) / 1024.0);
    std::fprintf(out, "[DeviceHwInfo] regs_per_block: %d\n", h.regs_per_block);
    std::fprintf(out, "[DeviceHwInfo] regs_per_multiprocessor: %d\n", h.regs_per_multiprocessor);
    std::fprintf(out, "[DeviceHwInfo] warp_size: %d\n", h.warp_size);
    {
        const int ws = h.warp_size > 0 ? h.warp_size : 1;
        const int warps_per_block_max = h.max_threads_per_block / ws;
        const int warps_per_sm_max = h.max_threads_per_multiprocessor / ws;
        std::fprintf(out, "[DeviceHwInfo] max_threads_per_block: %d (%d warps)\n", h.max_threads_per_block,
                     warps_per_block_max);
        std::fprintf(out, "[DeviceHwInfo] max_threads_per_multiprocessor: %d (%d warps)\n",
                     h.max_threads_per_multiprocessor, warps_per_sm_max);
    }
    std::fprintf(out, "[DeviceHwInfo] max_warps_per_multiprocessor (prop/threads): %d\n",
                 h.max_warps_per_multiprocessor_from_prop);
    std::fprintf(out, "[DeviceHwInfo] warp_schedulers_per_sm (uarch table, CC 8.x/9.x/10.x/12.x=4 when known): %d\n",
                 h.warp_schedulers_per_sm);
    std::fprintf(out,
                 "[DeviceHwInfo] max_resident_warps_per_warp_scheduler (uarch, Ampere-family=12; 4*12=48): %d\n",
                 h.max_resident_warps_per_warp_scheduler);
    std::fprintf(out,
                 "[DeviceHwInfo] max_instructions_issued_per_cycle_per_SM_theoretical (==schedulers, "
                 "rough upper bound, not from cuda API): %d\n",
                 h.max_instructions_issued_per_cycle_per_sm_theoretical);
    std::fprintf(out, "[DeviceHwInfo] fp32_cores_per_sm (table): %d\n", h.fp32_cores_per_sm);
    std::fprintf(out, "[DeviceHwInfo] peak_fp32_tflops_theoretical: %.6f\n",
                 static_cast<double>(h.peak_fp32_tflops_theoretical));
    // 数值 = 字节/s / 1e9，单位 GB/s（吉字节每秒）；不是 Gbps（吉比特每秒）。
    std::fprintf(out, "[DeviceHwInfo] peak_dram_theoretical: %.6f GB/s\n",
                 static_cast<double>(h.peak_dram_gbps_theoretical));

    if (h.fp32_cores_per_sm == 0) {
        std::fprintf(out,
                     "[DeviceHwInfo] note: fp32_cores_per_sm unknown for this CC; extend "
                     "device_fp32_cuda_cores_per_sm() to compute theoretical FP32 peak.\n");
    }
    std::fflush(out);
}

#endif  // DEVICE_HW_INFO_CUH_

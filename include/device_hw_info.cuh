#ifndef DEVICE_HW_INFO_CUH_
#define DEVICE_HW_INFO_CUH_

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

// NVIDIA Ampere GA102 Architecture Whitepaper Table 6（TU102 / GA100 / GA10x）dense 合计
// 返回每 SM 每时钟周期 FP16 Tensor Core 路径上的 dense FMA 条数（非 FLOPs）
inline __host__ int device_tc_fp16_dense_fma_per_sm_per_cycle(int cc_major, int cc_minor) {
    const int cc = cc_major * 10 + cc_minor;
    switch (cc) {
    case 75:
        return 512; // Turing TU102，Table 6
    case 80:
        return 1024; // GA100 SM，Table 6
    case 86:
    case 87:
    case 89:
        return 512; // GA10x SM（Jetson Orin = 8.7）；Ada 8.9 近似同量级，细分见白皮书
    default:
        return 0;
    }
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
    int fp32_cores_per_sm{
        0}; // SM 里大约有 128 条并行的 FP32 流水线, 每条流水线每时钟最多 1 次 FP32 FMA
    int tc_fp16_dense_fma_per_sm_per_cycle{
        0}; // SM 上 FP16 dense Tensor Core 每时钟能完成的 FMA 次数
    float peak_fp32_tflops_theoretical{0.f};
    float peak_tc_tfops_theoretical{0.f};
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

inline __host__ DeviceRooflineParams to_device_roofline_params(const DeviceHwInfo &h) {
    return DeviceRooflineParams{h.peak_fp32_tflops_theoretical, h.peak_dram_gbps_theoretical,
                                h.sm_count, h.cc_major, h.cc_minor};
}

inline __host__ DeviceHwInfo query_device_hw_info(int device_id = 0) {
    DeviceHwInfo h{};
    h.device_id = device_id;

    cudaDeviceProp prop{};
    const cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        std::snprintf(h.name, sizeof(h.name), "(cudaGetDeviceProperties: %s)",
                      cudaGetErrorString(err));
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
    h.max_resident_warps_per_warp_scheduler =
        device_max_resident_warps_per_warp_scheduler(h.cc_major, h.cc_minor);
    h.max_instructions_issued_per_cycle_per_sm_theoretical =
        device_max_instructions_issued_per_cycle_per_sm(h.cc_major, h.cc_minor);

    h.fp32_cores_per_sm = device_fp32_cuda_cores_per_sm(h.cc_major, h.cc_minor);
    h.tc_fp16_dense_fma_per_sm_per_cycle =
        device_tc_fp16_dense_fma_per_sm_per_cycle(h.cc_major, h.cc_minor);

    // FP32 整卡峰值 TFLOP/s：
    // sm_count × fp32_cores_per_sm × 2 × clockRate_kHz × 1000 / 1e12（×2：每周期 FMA=2 FLOPs）
    // Jetson Orin GA10B：4×128×2×1.02e9/1e12 ≈ 1.044 TFLOP/s
    if (h.fp32_cores_per_sm > 0 && h.sm_clock_khz > 0 && h.sm_count > 0) {
        const double sm_hz = static_cast<double>(h.sm_clock_khz) * 1000.0;
        h.peak_fp32_tflops_theoretical =
            static_cast<float>((static_cast<double>(h.sm_count) *
                                static_cast<double>(h.fp32_cores_per_sm) * 2.0 * sm_hz) /
                               1e12);
    }

    // FP16 TC dense 整卡峰值（GA102 Table 6 FMA/SM/c ×2）：
    // sm_count × tc_fma_per_sm_per_cycle × 2 × clockRate_kHz × 1000 / 1e12
    // Jetson Orin GA10B：4×512×2×1.02e9/1e12 ≈ 4.178 TFLOP/s
    if (h.tc_fp16_dense_fma_per_sm_per_cycle > 0 && h.sm_clock_khz > 0 && h.sm_count > 0) {
        const double sm_hz = static_cast<double>(h.sm_clock_khz) * 1000.0;
        h.peak_tc_tfops_theoretical = static_cast<float>(
            (static_cast<double>(h.sm_count) *
             static_cast<double>(h.tc_fp16_dense_fma_per_sm_per_cycle) * 2.0 * sm_hz) /
            1e12);
    }

    // DRAM 峰值 GB/s：
    // memoryClockRate_kHz × 1000 × (memoryBusWidth bits / 8) × 2 / 1e9（×2=DDR）
    // Jetson Orin GA10B：1020000 × 1000 × (128/8) × 2 / 1e9 = 32.64 GB/s
    if (h.mem_clock_khz > 0 && h.mem_bus_width_bits > 0) {
        const double byte_per_sec = static_cast<double>(h.mem_clock_khz) * 1000.0 *
                                    (static_cast<double>(h.mem_bus_width_bits) / 8.0) * 2.0;
        h.peak_dram_gbps_theoretical = static_cast<float>(byte_per_sec / 1e9);
    }

    return h;
}

inline __host__ void fprint_device_hw_info(FILE *out, const DeviceHwInfo &h) {
    if (!out) {
        return;
    }

    std::fprintf(out, "[DeviceHwInfo] device_id: %d\n", h.device_id);
    std::fprintf(out, "[DeviceHwInfo] name: %s\n", h.name);
    std::fprintf(out, "[DeviceHwInfo] compute_capability: %d.%d\n", h.cc_major, h.cc_minor);
    std::fprintf(out, "[DeviceHwInfo] sm_count: %d\n", h.sm_count);
    std::fprintf(out, "[DeviceHwInfo] sm_clock_ghz: %.6f\n",
                 static_cast<double>(h.sm_clock_khz) / 1e6);
    std::fprintf(out, "[DeviceHwInfo] mem_clock_ghz: %.6f\n",
                 static_cast<double>(h.mem_clock_khz) / 1e6);
    std::fprintf(out, "[DeviceHwInfo] mem_bus_width_bits: %d\n", h.mem_bus_width_bits);
    std::fprintf(out, "[DeviceHwInfo] total_global_mem_gbytes: %.6f\n",
                 static_cast<double>(h.total_global_mem_bytes) / 1e9);
    // 千字节：按 1024 字节 = 1 KiB（与 CUDA 内存规格习惯一致）
    std::fprintf(out, "[DeviceHwInfo] total_const_mem_kibytes: %.6f\n",
                 static_cast<double>(h.total_const_mem_bytes) / 1024.0);
    std::fprintf(out, "[DeviceHwInfo] l2_cache_bytes: %d\n", h.l2_cache_bytes);
    std::fprintf(out, "[DeviceHwInfo] global_l1_cache_supported: %d\n",
                 h.global_l1_cache_supported);
    std::fprintf(out, "[DeviceHwInfo] local_l1_cache_supported: %d\n", h.local_l1_cache_supported);
    std::fprintf(out, "[DeviceHwInfo] l1_data_cache_size_per_sm_bytes: N/A\n");
    std::fprintf(
        out,
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
        std::fprintf(out, "[DeviceHwInfo] max_threads_per_block: %d (%d warps)\n",
                     h.max_threads_per_block, warps_per_block_max);
        std::fprintf(out, "[DeviceHwInfo] max_threads_per_multiprocessor: %d (%d warps)\n",
                     h.max_threads_per_multiprocessor, warps_per_sm_max);
    }
    std::fprintf(out, "[DeviceHwInfo] max_warps_per_multiprocessor (prop/threads): %d\n",
                 h.max_warps_per_multiprocessor_from_prop);
    std::fprintf(out,
                 "[DeviceHwInfo] warp_schedulers_per_sm (uarch table, CC 8.x/9.x/10.x/12.x=4 when "
                 "known): %d\n",
                 h.warp_schedulers_per_sm);
    std::fprintf(out,
                 "[DeviceHwInfo] max_resident_warps_per_warp_scheduler (uarch, Ampere-family=12; "
                 "4*12=48): %d\n",
                 h.max_resident_warps_per_warp_scheduler);
    std::fprintf(
        out,
        "[DeviceHwInfo] max_instructions_issued_per_cycle_per_SM_theoretical (==schedulers, "
        "rough upper bound, not from cuda API): %d\n",
        h.max_instructions_issued_per_cycle_per_sm_theoretical);
    std::fprintf(out, "[DeviceHwInfo] fp32_cores_per_sm (table): %d\n", h.fp32_cores_per_sm);
    std::fprintf(out,
                 "[DeviceHwInfo] tc_fp16_dense_fma_per_sm_per_cycle (whitepaper Table 6): %d\n",
                 h.tc_fp16_dense_fma_per_sm_per_cycle);

    // 数值 = 字节/s / 1e9，单位 GB/s
    std::fprintf(out, "[DeviceHwInfo] peak_dram_theoretical: %.6f GB/s\n",
                 static_cast<double>(h.peak_dram_gbps_theoretical));
    std::fprintf(out, "[DeviceHwInfo] peak_fp32_tflops_theoretical: %.6f\n",
                 static_cast<double>(h.peak_fp32_tflops_theoretical));
    std::fprintf(out,
                 "[DeviceHwInfo] peak_tc_tfops_theoretical: %.6f  (FP16 Tensor dense; each FMA = 2 "
                 "FLOPs)\n",
                 static_cast<double>(h.peak_tc_tfops_theoretical));

    if (h.peak_fp32_tflops_theoretical > 0.f && h.peak_dram_gbps_theoretical > 0.f) {
        const double ridge_fp32 = (static_cast<double>(h.peak_fp32_tflops_theoretical) * 1e12) /
                                  (static_cast<double>(h.peak_dram_gbps_theoretical) * 1e9);
        std::fprintf(out, "[DeviceHwInfo] roofline_ridge_fp32_cuda_vs_dram_flops_per_byte: %.6f\n",
                     ridge_fp32);
    }

    if (h.tc_fp16_dense_fma_per_sm_per_cycle > 0 && h.peak_dram_gbps_theoretical > 0.f) {
        const double ridge_tc = (static_cast<double>(h.peak_tc_tfops_theoretical) * 1e12) /
                                (static_cast<double>(h.peak_dram_gbps_theoretical) * 1e9);
        std::fprintf(out,
                     "[DeviceHwInfo] roofline_ridge_tc_fp16_dense_vs_dram_flops_per_byte: %.6f\n",
                     ridge_tc);
    }

    if (h.fp32_cores_per_sm == 0) {
        std::fprintf(out, "[DeviceHwInfo] note: fp32_cores_per_sm unknown for this CC; extend "
                          "device_fp32_cuda_cores_per_sm() to compute theoretical FP32 peak.\n");
    }
    if (h.tc_fp16_dense_fma_per_sm_per_cycle == 0) {
        std::fprintf(
            out,
            "[DeviceHwInfo] note: tc_fp16_dense_fma_per_sm_per_cycle unknown for this CC; extend "
            "device_tc_fp16_dense_fma_per_sm_per_cycle() for peak_tc_tfops_theoretical.\n");
    }
    std::fflush(out);
}

#endif // DEVICE_HW_INFO_CUH_

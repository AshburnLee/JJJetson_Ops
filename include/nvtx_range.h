#pragma once

// NVTX 范围的 RAII 包装：构造时 push，析构时 pop。
// 放 include/，Model / Engine / GenerateLoop 都能直接 #include。
// 约定：最外层组件（create_model / create_engine）、最外层步骤
// （load_safetensors / load_fixture / load_weights / generate / prefill / decode），
// 以及 Engine 生产入口 forward_token_device（色块名 forward，谁调都有）。
// 不拆 q_linear / malloc / layer_N。
// 两端各钉一条瞬时 mark：engine_start（进 C++）/ engine_end（destroy_model 拆完）。
//
// 为什么不用手写 nvtxRangePushA / nvtxRangePop：
//   中间 return / 抛错时容易漏 pop，nsys 时间线会对不齐。
//   对象活多久，事件就有多长，跟 C++ 作用域绑死。
//
// 例（decode 一步，T=1）：
//   {
//     NvtxRange r("decode");          // push "decode"
//     ie_forward_token_sample(...);  // 这段里的 CUDA API / kernel 落在事件内
//   }                                 // 析构 pop
//   nsys 时间线上出现名为 decode 的一段 range。
//
// 默认关闭（空操作）。开启：cmake -DENABLE_NVTX=ON，或 ./build_all.sh --nvtx。
// nvtx3 是 header-only，不用链 libnvToolsExt。

// timeline 中需要不同事件显示不同颜色时，再添加
#ifdef ENABLE_NVTX
#include <cstdint>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <nvtx3/nvToolsExt.h>
#endif

// 主机侧只有一条 OS 线程在推 Engine（不另起 std::thread）。
// nsys 默认线程叫 python；profile 时改成 my_engine，方便从 CUPTI worker 里认出来。
// Linux 线程名最多 15 字符。ENABLE_NVTX 关闭时是空操作。
//
// 例：create_model 里调用一次后，nsys 线程行从 python 变成 my_engine。
#ifdef ENABLE_NVTX
// 瞬时 mark：没有时长。color 是 ARGB。
// 例：nvtx_mark_once("engine_start", 0xFF8000FF) -> nsys 上一条紫色竖线。
inline void nvtx_mark_once(const char *name, uint32_t color) {
    nvtxEventAttributes_t attr{};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = color;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    nvtxMarkEx(&attr);
}
#endif

inline void nvtx_mark_engine_begin() {
#ifdef ENABLE_NVTX
    // 进程里只钉一次：这条竖线左边是 nsys / import，右边才是我们的业务。
    // 不是 range。nsys 上搜 engine_start。
    static bool done = false;
    if (done) {
        return;
    }
    done = true;
    nvtx_mark_once("engine_start", 0xFF8000FF);
#endif
}

inline void nvtx_mark_engine_end() {
#ifdef ENABLE_NVTX
    // 进程里只钉一次：destroy_model 把权重槽 / RoPE / embed 都拆完之后。
    // 这条竖线左边是 create -> load -> generate -> destroy，右边是 Python 收尾。
    // 不是 range。nsys 上搜 engine_end。
    static bool done = false;
    if (done) {
        return;
    }
    done = true;
    // ARGB：不透明橙，跟 start 的紫好区分。
    nvtx_mark_once("engine_end", 0xFFFF8800);
#endif
}

inline void name_engine_thread() {
#ifdef ENABLE_NVTX
    prctl(PR_SET_NAME, "my_engine", 0, 0, 0);
    const uint32_t tid = static_cast<uint32_t>(syscall(SYS_gettid));
    nvtxNameOsThreadA(tid, "my_engine");
    nvtx_mark_engine_begin();
#endif
}

class NvtxRange {
  public:
    // name：nsys 上显示的色块名，必须是编译期或调用方保活的 C 字符串。
    explicit NvtxRange(const char *name) {
#ifdef ENABLE_NVTX
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }

    ~NvtxRange() {
#ifdef ENABLE_NVTX
        nvtxRangePop();
#endif
    }

    NvtxRange(const NvtxRange &) = delete;
    NvtxRange &operator=(const NvtxRange &) = delete;
    NvtxRange(NvtxRange &&) = delete;
    NvtxRange &operator=(NvtxRange &&) = delete;
};

// 懒得给局部变量起名时用。__LINE__ 保证同一函数里多处不会撞名。
//   NVTX_RANGE("prefill");
#define NVTX_RANGE_CONCAT2(a, b) a##b
#define NVTX_RANGE_CONCAT(a, b) NVTX_RANGE_CONCAT2(a, b)
#define NVTX_RANGE(name) const NvtxRange NVTX_RANGE_CONCAT(_nvtx_range_, __LINE__)(name)

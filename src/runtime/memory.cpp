#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/memory.h"
#include "mini_infer/runtime/memory_pool.h"
#include "runtime/cuda_check.h"
#include "utils/logger.h"

namespace mtensor {
namespace {

void FreeCPU(void* data) noexcept { std::free(data); }

class CPUAllocator : public Allocator {
public:
    DataPtr allocate(std::size_t bytes) const override {
        if (!bytes) return {nullptr, Device::kCPU, FreeCPU};
        auto* data = std::malloc(bytes);
        if (!data) throw std::bad_alloc{};
        return {data, Device::kCPU, FreeCPU};
    }
};

#ifdef ENABLE_CUDA
constexpr std::size_t kPoolAlignment = 256;

std::size_t AlignBlockSize(std::size_t bytes) {
    if (bytes > std::numeric_limits<std::size_t>::max() - (kPoolAlignment - 1)) {
        throw std::bad_alloc{};
    }
    return ((bytes + kPoolAlignment - 1) / kPoolAlignment) * kPoolAlignment;
}

void FreeCUDARaw(void* device_ptr) noexcept {
    if (device_ptr == nullptr) return;
    const cudaError_t status = cudaFree(device_ptr);
    if (status == cudaSuccess || status == cudaErrorCudartUnloading) return;
    try {
        logger.error(std::string("cudaFree failed: ") + cudaGetErrorString(status));
    } catch (...) {
        std::fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(status));
    }
}

class CUDAPool {
public:
    struct CachedBlock {
        void* ptr;
        cudaEvent_t ready;
    };

    ~CUDAPool() {
        alive_.store(false, std::memory_order_release);
        empty_cache();
    }

    static std::atomic<bool>& alive_flag() noexcept { return alive_; }

    DataPtr allocate(std::size_t bytes, DataDeleter deleter) {
        // 空请求不分配；其余请求按池的对齐粒度向上取整。
        if (bytes == 0) return {nullptr, Device::kCUDA, FreeCUDARaw};
        const std::size_t aligned = AlignBlockSize(bytes);
        void* ptr = nullptr;

        // 持锁查缓存：从最小够用的容量开始，最多接受取整后需求的两倍。
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++stats_.allocations;
            for (auto it = free_blocks_.lower_bound(aligned);
                 it != free_blocks_.end() && it->first - aligned <= aligned; ++it) {
                auto& blocks = it->second;
                // 同容量的块从后往前查，优先检查最近归还的块。
                for (std::size_t index = blocks.size(); index; --index) {
                    const CachedBlock block = blocks[index - 1];
                    // 默认流上最后一次使用结束后，才复用这块显存。
                    const cudaError_t ready = cudaEventQuery(block.ready);
                    if (ready == cudaErrorNotReady) continue;
                    CUDA_CHECK(ready);

                    // 旧 event 销毁失败时，撤销在用登记，保留缓存条目。
                    active_blocks_.emplace(block.ptr, it->first);
                    try {
                        CUDA_CHECK(cudaEventDestroy(block.ready));
                    } catch (...) {
                        active_blocks_.erase(block.ptr);
                        throw;
                    }
                    ptr = block.ptr;
                    const std::size_t block_size = it->first;
                    blocks.erase(blocks.begin() + static_cast<std::ptrdiff_t>(index - 1));
                    if (blocks.empty()) free_blocks_.erase(it);
                    stats_.cached_bytes -= block_size;
                    stats_.allocated_bytes += block_size;
                    ++stats_.cache_hits;
                    return {ptr, Device::kCUDA, deleter};
                }
            }
        }

        // 未命中时在锁外申请显存；显存不足则清缓存并重试一次。
        cudaError_t status = cudaMalloc(&ptr, aligned);
        if (status == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            empty_cache();
            status = cudaMalloc(&ptr, aligned);
        }
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDA error at cudaMalloc: ") + cudaGetErrorString(status));
        }

        // 登记成功后再交给 DataPtr；登记失败时直接释放新块。
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            active_blocks_.emplace(ptr, aligned);
            stats_.allocated_bytes += aligned;
        } catch (...) {
            FreeCUDARaw(ptr);
            throw;
        }
        return {ptr, Device::kCUDA, deleter};
    }

    void release(void* ptr) noexcept {
        // 空指针无需处理；池开始析构后，直接释放显存。
        if (ptr == nullptr) return;
        if (!alive_.load(std::memory_order_acquire)) {
            FreeCUDARaw(ptr);
            return;
        }

        cudaEvent_t ready = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // 按地址查块容量；未登记的块直接释放，不进缓存。
            const auto it = active_blocks_.find(ptr);
            if (it == active_blocks_.end()) {
                FreeCUDARaw(ptr);
                return;
            }

            // 撤掉在用记录，并扣除整块容量。
            const std::size_t block_size = it->second;
            active_blocks_.erase(it);
            stats_.allocated_bytes -= block_size;

            // 在默认流上记录 event，再把块和 event 一起放进对应容量的缓存桶。
            cudaError_t status = cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);
            if (status == cudaSuccess) status = cudaEventRecord(ready, nullptr);
            if (status == cudaSuccess) {
                try {
                    free_blocks_[block_size].push_back({ptr, ready});
                    stats_.cached_bytes += block_size;
                    return;
                } catch (...) {
                    // 缓存登记失败，交给锁外的清理路径。
                }
            }
        }

        // 未能放入缓存：销毁已创建的 event，直接释放显存。
        if (ready) (void)cudaEventDestroy(ready);
        FreeCUDARaw(ptr);
    }

    void empty_cache() noexcept {
        // 持锁取走当前的缓存块并清零缓存统计，不动在用块。
        std::map<std::size_t, std::vector<CachedBlock>> to_free;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            to_free.swap(free_blocks_);
            stats_.cached_bytes = 0;
        }

        // 在锁外逐块等待 event，再销毁 event、释放显存。
        for (auto& [size, ptrs] : to_free) {
            (void)size;
            for (const CachedBlock& block : ptrs) {
                (void)cudaEventSynchronize(block.ready);
                (void)cudaEventDestroy(block.ready);
                FreeCUDARaw(block.ptr);
            }
        }
    }

    CUDAMemoryStats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

private:
    static inline std::atomic<bool> alive_{true};
    mutable std::mutex mutex_;
    std::unordered_map<void*, std::size_t> active_blocks_;
    std::map<std::size_t, std::vector<CachedBlock>> free_blocks_;
    CUDAMemoryStats stats_;
};

CUDAPool& GetCUDAPool() {
    static CUDAPool pool;
    return pool;
}

void ReturnToCUDAPool(void* device_ptr) noexcept {
    if (device_ptr == nullptr) return;
    if (!CUDAPool::alive_flag().load(std::memory_order_acquire)) {
        FreeCUDARaw(device_ptr);
        return;
    }
    GetCUDAPool().release(device_ptr);
}

class CUDAAllocator : public Allocator {
public:
    DataPtr allocate(std::size_t bytes) const override {
        return GetCUDAPool().allocate(bytes, ReturnToCUDAPool);
    }
};
#endif

}  // namespace

const Allocator& GetAllocator(Device device) {
    if (device == Device::kCPU) {
        static const CPUAllocator allocator;
        return allocator;
    }
    if (device == Device::kCUDA) {
#ifdef ENABLE_CUDA
        static const CUDAAllocator allocator;
        return allocator;
#else
        throw std::logic_error("GetAllocator: CUDA support is disabled");
#endif
    }
    throw std::invalid_argument("GetAllocator: unsupported device");
}

void CopyBytes(void* dst, Device dst_device, const void* src, Device src_device,
               std::size_t bytes) {
    if (!bytes) return;
    if (dst_device == Device::kCPU && src_device == Device::kCPU) {
        std::memcpy(dst, src, bytes);
        return;
    }
#ifdef ENABLE_CUDA
    const auto kind = dst_device == Device::kCPU ? cudaMemcpyDeviceToHost :
                      src_device == Device::kCPU ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
#else
    throw std::logic_error("CopyBytes: CUDA support is disabled");
#endif
}

void EmptyCUDACache() {
#ifdef ENABLE_CUDA
    GetCUDAPool().empty_cache();
#endif
}

CUDAMemoryStats GetCUDAMemoryStats() {
#ifdef ENABLE_CUDA
    return GetCUDAPool().stats();
#else
    return {};
#endif
}

}  // namespace mtensor

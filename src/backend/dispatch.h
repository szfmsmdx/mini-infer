#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "mini_infer/tensor/device.h"

namespace mtensor {

class DeviceContext;
class Tensor;
DeviceContext& GetDeviceContext(Device device);

// 同一个算子槽位的 CPU 和 CUDA 实现使用相同的函数签名。
template <typename FnPtr>
struct KernelSlot {
    FnPtr cpu = nullptr;
    FnPtr cuda = nullptr;

    void register_fn(Device device, FnPtr fn) {
        if (!fn) throw std::invalid_argument("KernelSlot: null kernel");
        FnPtr* target = nullptr;
        if (device == Device::kCPU) target = &cpu;
        else if (device == Device::kCUDA) target = &cuda;
        else throw std::invalid_argument("KernelSlot: unsupported device");
        if (*target) throw std::logic_error("KernelSlot: duplicate registration");
        *target = fn;
    }

    template <typename... Args>
    void call(Device device, const char* op_name, Args&&... args) const {
        FnPtr fn = nullptr;
        if (device == Device::kCPU) fn = cpu;
        else if (device == Device::kCUDA) fn = cuda;
        else throw std::invalid_argument(std::string(op_name) + ": unsupported device");
        if (!fn) throw std::logic_error(std::string(op_name) + ": no backend for device");
        fn(GetDeviceContext(device), std::forward<Args>(args)...);
    }
};

using UnaryKernel = void (*)(DeviceContext&, const Tensor&, Tensor&);
using BinaryKernel = void (*)(DeviceContext&, const Tensor&, const Tensor&, Tensor&);
using BinaryBackwardKernel = void (*)(DeviceContext&, const Tensor&, const Tensor&,
                                      const Tensor&, Tensor&, Tensor&);
using UnaryBackwardKernel = void (*)(DeviceContext&, const Tensor&, const Tensor&, Tensor&);
using SoftmaxKernel = void (*)(DeviceContext&, const Tensor&, int, Tensor&);
using SoftmaxBackwardKernel = void (*)(DeviceContext&, const Tensor&, const Tensor&, int, Tensor&);
using SumDimKernel = void (*)(DeviceContext&, const Tensor&, int, Tensor&);
using MeanKernel = void (*)(DeviceContext&, const Tensor&, int, bool, Tensor&);
using MeanBackwardKernel = void (*)(DeviceContext&, const Tensor&, const Tensor&, int, Tensor&);
using ExtremaKernel = void (*)(DeviceContext&, const Tensor&, Tensor&, std::size_t, bool);
using ExtremaBackwardKernel = void (*)(DeviceContext&, const Tensor&, const Tensor&,
                                       const Tensor&, Tensor&, std::size_t);
using RoPEKernel = void (*)(DeviceContext&, const Tensor&, int, float, bool, int, Tensor&);
using TernaryBackwardKernel = void (*)(DeviceContext&, const Tensor&, const Tensor&,
                                        const Tensor&, Tensor&);
using AttentionKernel = void (*)(DeviceContext&, const Tensor&, const Tensor&, const Tensor&,
                                  int, Tensor&, Tensor&);
using AttentionBackwardKernel = void (*)(DeviceContext&, const Tensor&, const Tensor&,
                                          const Tensor&, int, const Tensor&, const Tensor&,
                                          Tensor&, Tensor&, Tensor&);
using NormKernel = void (*)(DeviceContext&, const Tensor&, const Tensor&, float, Tensor&);
using NormBackwardKernel = void (*)(DeviceContext&, const Tensor&, const Tensor&,
                                     const Tensor&, float, Tensor&, Tensor&);
using FillValueKernel = void (*)(DeviceContext&, Tensor&, const void*);
using FillTensorKernel = void (*)(DeviceContext&, Tensor&, const Tensor&);
using ZeroKernel = void (*)(DeviceContext&, Tensor&);

struct KernelTable {
    KernelSlot<BinaryKernel> add;
    KernelSlot<BinaryKernel> sub;
    KernelSlot<BinaryKernel> mul;
    KernelSlot<BinaryKernel> div;
    KernelSlot<UnaryKernel> neg;
    KernelSlot<UnaryKernel> sqrt;
    KernelSlot<UnaryKernel> rsqrt;
    KernelSlot<UnaryKernel> exp;
    KernelSlot<UnaryKernel> log;
    KernelSlot<UnaryKernel> sub_backward;
    KernelSlot<BinaryBackwardKernel> mul_backward;
    KernelSlot<BinaryBackwardKernel> div_backward;
    KernelSlot<UnaryKernel> neg_backward;
    KernelSlot<UnaryBackwardKernel> sqrt_backward;
    KernelSlot<UnaryBackwardKernel> rsqrt_backward;
    KernelSlot<UnaryBackwardKernel> exp_backward;
    KernelSlot<UnaryBackwardKernel> log_backward;
    KernelSlot<UnaryKernel> sigmoid;
    KernelSlot<UnaryKernel> silu;
    KernelSlot<SoftmaxKernel> softmax;
    KernelSlot<SoftmaxBackwardKernel> softmax_backward;
    KernelSlot<BinaryKernel> matmul;
    KernelSlot<BinaryBackwardKernel> matmul_backward;
    KernelSlot<BinaryKernel> embedding;
    KernelSlot<BinaryKernel> embedding_backward;
    KernelSlot<RoPEKernel> rope;
    KernelSlot<BinaryKernel> cross_entropy;
    KernelSlot<TernaryBackwardKernel> cross_entropy_backward;
    KernelSlot<AttentionKernel> causal_attention;
    KernelSlot<AttentionBackwardKernel> causal_attention_backward;
    KernelSlot<NormKernel> rms_norm;
    KernelSlot<NormBackwardKernel> rms_norm_backward;
    KernelSlot<UnaryKernel> copy_strided;
    KernelSlot<UnaryKernel> cast;
    KernelSlot<FillValueKernel> fill_value;
    KernelSlot<FillTensorKernel> fill_tensor;
    KernelSlot<ZeroKernel> zero;
    KernelSlot<UnaryKernel> sum_all;
    KernelSlot<SumDimKernel> sum_dim;
    KernelSlot<MeanKernel> mean;
    KernelSlot<MeanBackwardKernel> mean_backward;
    KernelSlot<ExtremaKernel> extrema;
    KernelSlot<ExtremaBackwardKernel> extrema_backward;
    KernelSlot<UnaryBackwardKernel> sigmoid_backward;
    KernelSlot<UnaryBackwardKernel> silu_backward;
};

const KernelTable& GetKernelTable();
void RegisterCPUKernels(KernelTable& table);
void RegisterCUDAKernels(KernelTable& table);

}  // namespace mtensor

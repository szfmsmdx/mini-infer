#include "backend/dispatch.h"
#include "mini_infer/tensor/tensor.h"
#include "ops/activation/kernels.h"
#include "ops/elementwise/kernels.h"
#include "ops/copy/kernels.h"
#include "ops/fill/kernels.h"
#include "ops/matmul/kernels.h"
#include "ops/reduction/kernels.h"

namespace mtensor {
void EmbeddingCPU(const Tensor& weight, const Tensor& indices, Tensor& output);
void EmbeddingGradCPU(const Tensor& grad, const Tensor& indices, Tensor& weight_grad);
void RoPECPU(const Tensor& x, int head_width, float base, bool inverse,
             int pos_offset, Tensor& output);
void CausalAttentionCPU(const Tensor& q, const Tensor& k, const Tensor& v, int heads,
                        Tensor& prob, Tensor& output);
void CausalAttentionGradCPU(const Tensor& q, const Tensor& k, const Tensor& v, int heads,
                            const Tensor& prob, const Tensor& grad,
                            Tensor& dq, Tensor& dk, Tensor& dv);
void RMSNormCPU(const Tensor& x, const Tensor& weight, float eps, Tensor& output);
void RMSNormBackwardCPU(const Tensor& x, const Tensor& weight, const Tensor& grad,
                        float eps, Tensor& dx, Tensor& dw);
}

namespace mtensor {

void RegisterCPUKernels(KernelTable& table) {
    table.add.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& a, const Tensor& b, Tensor& out) {
            AddCPU(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.dtype(),
                   out.shape(), a.stride(), b.stride(), out.stride());
        });
    table.sub.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& a, const Tensor& b, Tensor& out) {
            SubCPU(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.dtype(),
                   out.shape(), a.stride(), b.stride(), out.stride());
        });
    table.mul.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& a, const Tensor& b, Tensor& out) {
            MulCPU(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.dtype(),
                   out.shape(), a.stride(), b.stride(), out.stride());
        });
    table.div.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& a, const Tensor& b, Tensor& out) {
            DivCPU(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.dtype(),
                   out.shape(), a.stride(), b.stride(), out.stride());
        });
    table.neg.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& x, Tensor& out) {
            NegCPU(x.data_ptr(), out.data_ptr(), x.dtype(),
                   x.shape(), x.stride(), out.stride());
        });
    table.sqrt.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& x, Tensor& out) {
            SqrtCPU(x.data_ptr(), out.data_ptr(), x.dtype(), x.shape(), x.stride());
        });
    table.rsqrt.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& x, Tensor& out) { RsqrtCPU(x, out); });
    table.exp.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& x, Tensor& out) { ExpCPU(x, out); });
    table.log.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& x, Tensor& out) { LogCPU(x, out); });
    table.rsqrt_backward.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& x, const Tensor& grad, Tensor& dx) {
            RsqrtBackwardCPU(x, grad, dx);
        });
    table.sigmoid.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& x, Tensor& out) { SigmoidCPU(x, out); });
    table.silu.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& x, Tensor& out) { SiluCPU(x, out); });
    table.sigmoid_backward.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& x, const Tensor& grad, Tensor& dx) {
            SigmoidBackwardCPU(x, grad, dx);
        });
    table.silu_backward.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& x, const Tensor& grad, Tensor& dx) {
            SiluBackwardCPU(x, grad, dx);
        });
    table.matmul.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& a, const Tensor& b, Tensor& out) {
            MatmulCPU(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.dtype(),
                      a.size(0), a.size(1), b.size(1), a.stride(), b.stride());
        });
    table.embedding.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& weight, const Tensor& indices, Tensor& out) {
            EmbeddingCPU(weight, indices, out);
        });
    table.embedding_backward.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& grad, const Tensor& indices, Tensor& dw) {
            EmbeddingGradCPU(grad, indices, dw);
        });
    table.rope.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& x, int head_width, float base,
            bool inverse, int pos_offset, Tensor& out) {
            RoPECPU(x, head_width, base, inverse, pos_offset, out);
        });
    table.causal_attention.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& q, const Tensor& k, const Tensor& v,
            int heads, Tensor& prob, Tensor& out) {
            CausalAttentionCPU(q, k, v, heads, prob, out);
        });
    table.causal_attention_backward.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& q, const Tensor& k, const Tensor& v, int heads,
            const Tensor& prob, const Tensor& grad,
            Tensor& dq, Tensor& dk, Tensor& dv) {
            CausalAttentionGradCPU(q, k, v, heads, prob, grad, dq, dk, dv);
        });
    table.rms_norm.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& x, const Tensor& weight, float eps, Tensor& out) {
            RMSNormCPU(x, weight, eps, out);
        });
    table.rms_norm_backward.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& x, const Tensor& weight,
            const Tensor& grad, float eps, Tensor& dx, Tensor& dw) {
            RMSNormBackwardCPU(x, weight, grad, eps, dx, dw);
        });
    table.copy_strided.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& src, Tensor& dst) {
            CopyCPU(dst.data_ptr(), src.data_ptr(), dst.shape(), dst.stride(),
                    src.stride(), dst.element_size());
        });
    table.cast.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& src, Tensor& dst) { CastCPU(src, dst); });
    table.fill_value.register_fn(Device::kCPU,
        +[](DeviceContext&, Tensor& dst, const void* value) {
            FillCPU(dst.data_ptr(), dst.shape(), dst.stride(), dst.element_size(), value);
        });
    table.fill_tensor.register_fn(Device::kCPU,
        +[](DeviceContext&, Tensor& dst, const Tensor& value) {
            FillCPU(dst.data_ptr(), dst.shape(), dst.stride(),
                    dst.element_size(), value.data_ptr());
        });
    table.zero.register_fn(Device::kCPU,
        +[](DeviceContext&, Tensor& dst) {
            ZeroCPU(dst.data_ptr(), dst.shape(), dst.stride(), dst.element_size());
        });
    table.sum_all.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& x, Tensor& out) {
            SumCPU(x.data_ptr(), out.data_ptr(), x.dtype(),
                   x.shape(), x.stride(), x.numel());
        });
    table.sum_dim.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& x, int dim, Tensor& out) {
            SumCPU(x.data_ptr(), out.data_ptr(), x.dtype(),
                   x.shape(), x.stride(), x.numel(), dim);
        });
    table.extrema.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& x, Tensor& out,
            std::size_t inner_size, bool maximum) {
            ExtremaCPU(x, out, inner_size, maximum);
        });
    table.extrema_backward.register_fn(Device::kCPU,
        +[](DeviceContext&, const Tensor& x, const Tensor& extrema,
            const Tensor& grad, Tensor& dx, std::size_t inner_size) {
            ExtremaGradCPU(x, extrema, grad, dx, inner_size);
        });
}

}  // namespace mtensor

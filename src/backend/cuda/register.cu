#include "backend/dispatch.h"
#include "mini_infer/tensor/tensor.h"
#include "ops/activation/kernels.h"
#include "ops/elementwise/kernels.h"
#include "ops/copy/kernels.h"
#include "ops/fill/kernels.h"
#include "ops/matmul/kernels.h"
#include "ops/reduction/kernels.h"
#include "ops/norm/kernels.h"
#include "backend/common/cuda_layout.h"

namespace mtensor {
void EmbeddingCUDA(const Tensor& weight, const Tensor& indices, Tensor& output);
void EmbeddingGradCUDA(const Tensor& grad, const Tensor& indices, Tensor& weight_grad);
void RoPECUDA(const Tensor& x, int head_width, float base, bool inverse,
              int pos_offset, Tensor& output);
void CrossEntropyCUDA(const Tensor& logits, const Tensor& targets, Tensor& row_losses);
void CrossEntropyGradCUDA(const Tensor& logits, const Tensor& targets,
                          const Tensor& grad, Tensor& logits_grad);
void CausalAttentionCUDA(const Tensor& q, const Tensor& k, const Tensor& v, int heads,
                         Tensor& prob, Tensor& output);
void CausalAttentionGradCUDA(const Tensor& q, const Tensor& k, const Tensor& v, int heads,
                             const Tensor& prob, const Tensor& grad,
                             Tensor& dq, Tensor& dk, Tensor& dv);
}

namespace mtensor {

void RegisterCUDAKernels(KernelTable& table) {
    table.add.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& a, const Tensor& b, Tensor& out) {
            AddCUDA(a.data_ptr(), b.data_ptr(), out.data_ptr(), out.numel(), a.dtype(),
                    PackLayout(a), PackLayout(b));
        });
    table.sub.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& a, const Tensor& b, Tensor& out) {
            SubCUDA(a.data_ptr(), b.data_ptr(), out.data_ptr(), out.numel(), a.dtype(),
                    PackLayout(a), PackLayout(b));
        });
    table.mul.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& a, const Tensor& b, Tensor& out) {
            MulCUDA(a.data_ptr(), b.data_ptr(), out.data_ptr(), out.numel(), a.dtype(),
                    PackLayout(a), PackLayout(b));
        });
    table.div.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& a, const Tensor& b, Tensor& out) {
            DivCUDA(a.data_ptr(), b.data_ptr(), out.data_ptr(), out.numel(), a.dtype(),
                    PackLayout(a), PackLayout(b));
        });
    table.neg.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& x, Tensor& out) {
            NegCUDA(x.data_ptr(), out.data_ptr(), x.numel(), x.dtype(), PackLayout(x));
        });
    table.sqrt.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& x, Tensor& out) {
            SqrtCUDA(x.data_ptr(), out.data_ptr(), x.dtype(), x.numel(), PackLayout(x));
        });
    table.rsqrt.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& x, Tensor& out) { RsqrtCUDA(x, out); });
    table.exp.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& x, Tensor& out) { ExpCUDA(x, out); });
    table.log.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& x, Tensor& out) { LogCUDA(x, out); });
    table.rsqrt_backward.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& x, const Tensor& grad, Tensor& dx) {
            RsqrtBackwardCUDA(x, grad, dx);
        });
    table.sigmoid.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& x, Tensor& out) { SigmoidCUDA(x, out); });
    table.silu.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& x, Tensor& out) { SiluCUDA(x, out); });
    table.sigmoid_backward.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& x, const Tensor& grad, Tensor& dx) {
            SigmoidBackwardCUDA(x, grad, dx);
        });
    table.silu_backward.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& x, const Tensor& grad, Tensor& dx) {
            SiluBackwardCUDA(x, grad, dx);
        });
    table.matmul.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& a, const Tensor& b, Tensor& out) {
            MatmulCUDA(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.dtype(),
                       a.size(0), a.size(1), b.size(1), a.stride(), b.stride());
        });
    table.embedding.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& weight, const Tensor& indices, Tensor& out) {
            EmbeddingCUDA(weight, indices, out);
        });
    table.embedding_backward.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& grad, const Tensor& indices, Tensor& dw) {
            EmbeddingGradCUDA(grad, indices, dw);
        });
    table.rope.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& x, int head_width, float base,
            bool inverse, int pos_offset, Tensor& out) {
            RoPECUDA(x, head_width, base, inverse, pos_offset, out);
        });
    table.cross_entropy_rows.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& logits, const Tensor& targets, Tensor& rows) {
            CrossEntropyCUDA(logits, targets, rows);
        });
    table.cross_entropy_backward.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& logits, const Tensor& targets,
            const Tensor& grad, Tensor& dx) {
            CrossEntropyGradCUDA(logits, targets, grad, dx);
        });
    table.causal_attention.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& q, const Tensor& k, const Tensor& v,
            int heads, Tensor& prob, Tensor& out) {
            CausalAttentionCUDA(q, k, v, heads, prob, out);
        });
    table.causal_attention_backward.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& q, const Tensor& k, const Tensor& v, int heads,
            const Tensor& prob, const Tensor& grad,
            Tensor& dq, Tensor& dk, Tensor& dv) {
            CausalAttentionGradCUDA(q, k, v, heads, prob, grad, dq, dk, dv);
        });
    table.rms_norm.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& x, const Tensor& weight, float eps, Tensor& out) {
            RMSNormCUDA(x, weight, eps, out);
        });
    table.rms_norm_backward.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& x, const Tensor& weight,
            const Tensor& grad, float eps, Tensor& dx, Tensor& dw) {
            RMSNormBackwardCUDA(x, weight, grad, eps, dx, dw);
        });
    table.copy_strided.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& src, Tensor& dst) {
            CopyCUDA(dst.data_ptr(), src.data_ptr(), PackLayout(dst), PackLayout(src),
                     dst.numel(), dst.element_size());
        });
    table.cast.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& src, Tensor& dst) { CastCUDA(src, dst); });
    table.fill_value.register_fn(Device::kCUDA,
        +[](DeviceContext&, Tensor& dst, const void* value) {
            FillCUDA(dst.data_ptr(), dst.numel(), dst.dtype(), value, PackLayout(dst));
        });
    table.fill_tensor.register_fn(Device::kCUDA,
        +[](DeviceContext&, Tensor& dst, const Tensor& value) {
            FillCUDAFromDevice(dst.data_ptr(), dst.numel(), dst.element_size(),
                               value.data_ptr(), PackLayout(dst));
        });
    table.zero.register_fn(Device::kCUDA,
        +[](DeviceContext&, Tensor& dst) {
            ZeroCUDA(dst.data_ptr(), PackLayout(dst), dst.numel(), dst.element_size());
        });
    table.sum_all.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& x, Tensor& out) {
            SumCUDA(x.data_ptr(), out.data_ptr(), x.dtype(), PackLayout(x), x.numel());
        });
    table.sum_dim.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& x, int dim, Tensor& out) {
            SumCUDA(x.data_ptr(), out.data_ptr(), x.dtype(), PackLayout(x), out.numel(), dim);
        });
    table.extrema.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& x, Tensor& out,
            std::size_t inner_size, bool maximum) {
            ExtremaCUDA(x, out, inner_size, maximum);
        });
    table.extrema_backward.register_fn(Device::kCUDA,
        +[](DeviceContext&, const Tensor& x, const Tensor& extrema,
            const Tensor& grad, Tensor& dx, std::size_t inner_size) {
            ExtremaGradCUDA(x, extrema, grad, dx, inner_size);
        });
}

}  // namespace mtensor

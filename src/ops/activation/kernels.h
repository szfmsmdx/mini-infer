#pragma once

namespace mtensor {

class Tensor;

void SigmoidCPU(const Tensor& x, Tensor& out);
void SiluCPU(const Tensor& x, Tensor& out);
void SigmoidBackwardCPU(const Tensor& x, const Tensor& output_grad, Tensor& dx);
void SiluBackwardCPU(const Tensor& x, const Tensor& output_grad, Tensor& dx);
// Softmax 的前后向统一使用 FP32 张量，调用层负责 dtype 转换和关闭建图。
void SoftmaxCPU(const Tensor& x, int dim, Tensor& prob);
void SoftmaxBackwardCPU(const Tensor& prob, const Tensor& output_grad, int dim, Tensor& dx);

void SigmoidCUDA(const Tensor& x, Tensor& out);
void SiluCUDA(const Tensor& x, Tensor& out);
void SigmoidBackwardCUDA(const Tensor& x, const Tensor& output_grad, Tensor& dx);
void SiluBackwardCUDA(const Tensor& x, const Tensor& output_grad, Tensor& dx);
void SoftmaxCUDA(const Tensor& x, int dim, Tensor& prob);
void SoftmaxBackwardCUDA(const Tensor& prob, const Tensor& output_grad, int dim, Tensor& dx);

}  // namespace mtensor

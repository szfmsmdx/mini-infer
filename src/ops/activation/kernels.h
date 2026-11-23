#pragma once

namespace mtensor {

class Tensor;

void SigmoidCPU(const Tensor& x, Tensor& out);
void SiluCPU(const Tensor& x, Tensor& out);
void SigmoidBackwardCPU(const Tensor& x, const Tensor& output_grad, Tensor& dx);
void SiluBackwardCPU(const Tensor& x, const Tensor& output_grad, Tensor& dx);

void SigmoidCUDA(const Tensor& x, Tensor& out);
void SiluCUDA(const Tensor& x, Tensor& out);
void SigmoidBackwardCUDA(const Tensor& x, const Tensor& output_grad, Tensor& dx);
void SiluBackwardCUDA(const Tensor& x, const Tensor& output_grad, Tensor& dx);

}  // namespace mtensor

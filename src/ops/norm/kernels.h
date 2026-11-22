#pragma once

namespace mtensor {

class Tensor;

void RMSNormCUDA(const Tensor& x, const Tensor& weight, float eps, Tensor& output);

void RMSNormBackwardCUDA(const Tensor& x, const Tensor& weight, const Tensor& output_grad,
                         float eps, Tensor& dx, Tensor& dw);

}  // namespace mtensor

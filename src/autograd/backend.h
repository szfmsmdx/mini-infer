#pragma once

namespace mtensor {

class Tensor;
void RsqrtBackwardCPU(const Tensor& x, const Tensor& grad, Tensor& dx);
void RsqrtBackwardCUDA(const Tensor& x, const Tensor& grad, Tensor& dx);

}  // namespace mtensor

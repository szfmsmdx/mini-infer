#include <cstddef>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "mini_infer/autograd.h"
#include "mini_infer/ops/elementwise.h"
#include "mini_infer/tensor.h"

namespace {
using namespace mtensor;

void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void CheckThrows(Function function) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error("expected exception");
}

void TestPermute(const std::vector<int>& dims) {
    const std::vector<int> shape = {2, 3, 4, 5};
    std::vector<float> values(120);
    std::iota(values.begin(), values.end(), 0.0f);
    Tensor x(values.data(), shape, Dtype::kFloat32);
    x.requires_grad_(true);
    Tensor y = x.permute(dims);
    Check(y.data_ptr() == x.data_ptr(), "permute must share storage");
    for (std::size_t i = 0; i < dims.size(); ++i) {
        Check(y.shape()[i] == x.shape()[dims[i]], "permuted shape");
        Check(y.stride()[i] == x.stride()[dims[i]], "permuted stride");
    }
    Check(x.shape() == shape, "permute changed input shape");
    Check(y.grad_node() != nullptr, "permute must create a node");

    std::vector<float> grad_values(120);
    std::iota(grad_values.begin(), grad_values.end(), 1.0f);
    Tensor output_grad(grad_values.data(), y.shape(), Dtype::kFloat32);
    NogradNode guard;
    auto input_grads = y.grad_node()->backward(output_grad);
    Check(input_grads.size() == 1, "permute gradient count");
    const Tensor& input_grad = input_grads[0];
    Check(input_grad.shape() == shape, "permute gradient shape");
    Check(!input_grad.grad_node(), "backward must not create a node under no-grad");
    const float* grad_data = static_cast<const float*>(input_grad.data_ptr());
    for (int a = 0; a < shape[0]; ++a)
        for (int b = 0; b < shape[1]; ++b)
            for (int c = 0; c < shape[2]; ++c)
                for (int d = 0; d < shape[3]; ++d) {
                    const int indices[] = {a, b, c, d};
                    std::size_t input_offset = 0, output_offset = 0;
                    for (std::size_t i = 0; i < dims.size(); ++i) {
                        input_offset += indices[i] * input_grad.stride()[i];
                        output_offset += indices[dims[i]] * output_grad.stride()[i];
                    }
                    Check(grad_data[input_offset] == grad_values[output_offset],
                          "permuted gradient element");
                }
}

void TestErrorsAndBackward() {
    Tensor x({2, 3, 4}, Dtype::kFloat32);
    CheckThrows<std::invalid_argument>([&] { x.permute({0, 1}); });
    CheckThrows<std::invalid_argument>([&] { x.permute({0, 0, 2}); });
    CheckThrows<std::out_of_range>([&] { x.permute({0, -1, 2}); });
    CheckThrows<std::out_of_range>([&] { x.permute({0, 1, 3}); });

    float value = 3.0f;
    Tensor scalar(&value, {1, 1, 1}, Dtype::kFloat32);
    scalar.requires_grad_(true);
    Tensor y = scalar.permute({2, 0, 1});
    mul(y, y).backward();
    Check(scalar.has_grad(), "missing end-to-end gradient");
    Check(*static_cast<const float*>(scalar.grad().data_ptr()) == 6.0f,
          "incorrect end-to-end gradient");
}
}  // namespace

int main() {
    try {
        TestPermute({0, 1, 2, 3});
        TestPermute({1, 0, 2, 3});
        TestPermute({1, 2, 3, 0});
        TestPermute({3, 1, 0, 2});
        TestErrorsAndBackward();
        std::cout << "CPU layout tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

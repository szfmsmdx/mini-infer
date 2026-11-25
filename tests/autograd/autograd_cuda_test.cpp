#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mini_infer/autograd.h"
#include "ops/copy/nodes.h"
#include "mini_infer/ops/elementwise.h"
#include "mini_infer/tensor/tensor.h"

namespace {
using namespace mtensor;

void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void CheckScalarGradient(const Tensor& tensor, float expected) {
    Check(tensor.has_grad(), "missing gradient");
    NogradNode guard;
    Tensor cpu_grad = tensor.grad().to(Device::kCPU);
    float actual = *static_cast<const float*>(cpu_grad.data_ptr());
    Check(std::fabs(actual - expected) < 1e-5f, "incorrect gradient value");
    Check(tensor.grad().device() == tensor.device(), "gradient is on wrong device");
}

void TestInplaceVersions() {
    auto require = [](bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    };
    auto rejects_modified = [&require](Tensor loss) {
        bool rejected = false;
        try { loss.backward(); }
        catch (const std::logic_error& error) {
            rejected = std::string(error.what()).find("input modified after forward") != std::string::npos;
        }
        require(rejected && EnableGrad(), "backward must reject changed saved data and restore grad mode");
    };
    Tensor base = full({2, 3}, 3.0f, Dtype::kFloat32, Device::kCUDA);
    std::vector<Tensor> aliases = {
        base, base[0], base.narrow(1, 1, 2), base.transpose(0, 1), base.permute({1, 0}),
        base.reshape({6}), base.unsqueeze(1), base.unsqueeze(1).squeeze(1), base.detach(),
        base.to(Device::kCUDA), base.narrow(0, 0, 1).broadcast({2, 3})
    };
    auto version = base.version();
    aliases[1].fill_(5);
    require(base.version() == version + 1, "index write must update base version once");
    for (const auto& alias : aliases)
        require(alias.version() == base.version(), "all storage aliases must share the version");
    Tensor assigned;
    assigned = base;
    Tensor moved(std::move(assigned));
    Tensor destination;
    destination = std::move(moved);
    version = base.version();
    destination.zero_();
    require(base.version() == version + 1 && destination.version() == base.version(), "copy and move must retain version sharing");
    Tensor independent = base.clone();
    Tensor reshaped_copy = base.transpose(0, 1).reshape({6});
    independent.fill_(2);
    reshaped_copy.fill_(2);
    require(base.version() == version + 1, "independent allocations must not change base version");
    if (base.device() == Device::kCUDA) {
        Tensor cpu_copy = base.to(Device::kCPU);
        cpu_copy.zero_();
        require(base.version() == version + 1, "cross-device copy must have independent version");
    }
    version = base.version();
    base.copy_(base);
    require(base.version() == version, "self copy must not update version");
    bool rejected = false;
    try { base.copy_(full({1}, 0.0f, Dtype::kFloat32, Device::kCUDA)); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && base.version() == version, "failed copy validation must not update version");
    rejected = false;
    try { aliases.back().fill_(1); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && base.version() == version, "rejected broadcast write must not update version");
    Tensor scalar = full({}, 2.0f, Dtype::kFloat32, Device::kCUDA);
    base.fill_(scalar);
    require(base.version() == version + 1, "Tensor scalar fill must update version once");
    version = base.version();
    {
        NogradNode guard;
        base.fill_(base[0][0]);  // Scalar and destination share storage.
    }
    require(base.version() == version + 1, "aliased scalar fill must update version once");

    using BinaryOp = Tensor (*)(const Tensor&, const Tensor&);
    for (BinaryOp operation : {mul, div}) {
        for (Dtype dtype : {Dtype::kFloat32, Dtype::kFloat16, Dtype::kBFloat16}) {
            for (int side = 0; side < 2; ++side) {
                for (int write = 0; write < 4; ++write) {
                    Tensor a = full({2, 3}, 6, dtype, Device::kCUDA);
                    Tensor b = full({3}, 2, dtype, Device::kCUDA);
                    a.requires_grad_(true);
                    // b does not request a gradient, but its value is still needed for a's gradient.
                    Tensor loss = operation(a, b).sum();
                    Tensor target = side == 0 ? a : b;
                    auto before = target.version();
                    {
                        NogradNode guard;
                        if (write == 0) target.fill_(4);
                        else if (write == 1) target.zero_();
                        else if (write == 2) target.copy_(full(target.shape(), 4, dtype, Device::kCUDA));
                        else target.detach().narrow(0, 0, 1).fill_(4);
                    }
                    require(target.version() == before + 1, "in-place write must update shared version once");
                    rejects_modified(loss);
                }
            }
        }
    }
    Tensor x = full({2, 3}, 3.0f, Dtype::kFloat32, Device::kCUDA);
    x.requires_grad_(true);
    Tensor original_loss = mul(x, x).sum();
    // Modifying clone storage must not invalidate the original graph.
    x.clone().fill_(9);
    original_loss.backward();
    version = x.version();
    auto gradient_version = x.grad().version();
    x.zero_grad();
    require(x.version() == version && x.grad().version() == gradient_version + 1,
            "zero_grad changes gradient version only");
    original_loss.backward();  // Reusing an unchanged graph is still valid.
    {
        NogradNode guard;
        Tensor grad = x.grad().to(Device::kCPU);
        for (int i = 0; i < 6; ++i)
            require(static_cast<const float*>(grad.data_ptr())[i] == 6, "unchanged graph gradient");
        x.fill_(4);           // Parameter update after backward.
    }
    rejects_modified(original_loss);
    x.zero_grad();
    mul(x, x).sum().backward();  // New forward records the new version.
    {
        NogradNode guard;
        Tensor grad = x.grad().to(Device::kCPU);
        for (int i = 0; i < 6; ++i)
            require(static_cast<const float*>(grad.data_ptr())[i] == 8, "new graph after parameter update");
    }
    x.zero_grad();
    Tensor value_independent = sub(add(x, x), x).reshape({6}).sum();
    x.fill_(7);
    value_independent.backward();  // These derivatives do not read saved input data.
    {
        NogradNode guard;
        Tensor grad = x.grad().to(Device::kCPU);
        for (int i = 0; i < 6; ++i)
            require(static_cast<const float*>(grad.data_ptr())[i] == 1, "value-independent backward must not reject mutation");
    }
    Tensor external_loss = mul(x, x).sum();
    x.update_version();  // External writers explicitly notify the framework.
    rejects_modified(external_loss);
}
void TestZeroGrad() {
    auto require = [](bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    };
    Tensor empty;
    empty.zero_grad();
    require(!empty.has_grad(), "zero_grad must not create gradient on empty tensor");
    for (Dtype dtype : {Dtype::kFloat32, Dtype::kFloat16, Dtype::kBFloat16}) {
        auto read = [dtype](const Tensor& tensor, int i) {
            if (dtype == Dtype::kFloat32) return static_cast<const float*>(tensor.data_ptr())[i];
            const auto* bits = static_cast<const std::uint16_t*>(tensor.data_ptr()) + i;
            return dtype == Dtype::kFloat16 ? DecodeFloat16(bits) : DecodeBFloat16(bits);
        };
        auto check_values = [&require, &read](const Tensor& tensor, float expected) {
            NogradNode guard;
            Tensor host = tensor.to(Device::kCPU);
            for (std::size_t i = 0; i < host.numel(); ++i)
                require(read(host, i) == expected, "zero_grad gradient value");
        };
        Tensor x = full({2}, 3, dtype, Device::kCUDA);
        x.requires_grad_(true);
        x.zero_grad();
        require(!x.has_grad() && x.requires_grad(), "zero_grad on fresh leaf must be a no-op");
        Tensor output = mul(x, x);
        output.requires_grad_(true);
        auto node = output.grad_node();
        Tensor gradient = full({2}, 1, dtype, Device::kCUDA);
        output.backward(gradient);
        output.backward(gradient);
        check_values(x.grad(), 12);
        check_values(output.grad(), 2);
        Tensor saved_gradient = x.grad();
        const void* saved_pointer = saved_gradient.data_ptr();
        Tensor alias = x;
        alias.zero_grad();
        require(x.has_grad() && alias.has_grad() && x.grad().data_ptr() == saved_pointer,
                "zero_grad must retain shared gradient storage");
        require(x.requires_grad() && !x.grad_node() && output.grad_node() == node,
                "zero_grad must preserve gradient settings and graph");
        require(x.grad().shape() == x.shape() && x.grad().dtype() == dtype &&
                x.grad().device() == x.device(), "zero_grad must preserve gradient metadata");
        check_values(saved_gradient, 0);
        check_values(output.grad(), 2);  // Only the selected Tensor is cleared.
        check_values(x, 3);              // Parameter data is unchanged.
        x.zero_grad();
        check_values(x.grad(), 0);       // Repeated calls remain valid.
        output.zero_grad();
        require(output.has_grad() && output.requires_grad() && output.grad_node() == node,
                "zero_grad must preserve non-leaf metadata");
        check_values(output.grad(), 0);
        output.backward(gradient);
        check_values(x.grad(), 6);
        check_values(output.grad(), 1);
        require(!x.grad().grad_node() && !output.grad().grad_node(), "zero_grad must not record graph");
        require(EnableGrad(), "zero_grad must preserve grad mode");
        {
            NogradNode guard;
            x.zero_grad();
            require(!EnableGrad(), "zero_grad must preserve outer no-grad mode");
        }
        check_values(x.grad(), 0);
    }
    Tensor source = full({}, 1.0f, Dtype::kFloat32, Device::kCUDA);
    Tensor moved = std::move(source);
    source.zero_grad();
    require(!source.has_grad() && !moved.has_grad(), "zero_grad on moved-from tensor must be safe");
}
void TestExplicitBackwardGradient() {
    auto require = [](bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    };
    auto invalid = [&require](Tensor output, const Tensor& gradient) {
        bool rejected = false;
        try { output.backward(gradient); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && EnableGrad(), "invalid backward gradient must be rejected");
    };
    for (Dtype dtype : {Dtype::kFloat32, Dtype::kFloat16, Dtype::kBFloat16}) {
        Tensor x = full({2, 3}, 2, dtype, Device::kCUDA);
        x.requires_grad_(true);
        Tensor output = add(mul(x, x), x);
        output.requires_grad_(true);
        Tensor values = full({3, 2}, 0, dtype, Device::kCUDA);
        {
            NogradNode guard;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 2; ++j) values[i][j].fill_(i * 2 + j + 1);
        }
        Tensor gradient = values.transpose(0, 1);
        require(!gradient.is_contiguous(), "test requires strided gradient");
        output.backward(gradient);
        output.backward(gradient);
        require(EnableGrad(), "explicit backward must restore grad mode");
        {
            NogradNode guard;
            Tensor grad = x.grad().to(Device::kCPU);
            Tensor root_grad = output.grad().to(Device::kCPU);
            Tensor seed = gradient.clone().to(Device::kCPU);
            const float weights[] = {1, 3, 5, 2, 4, 6};
            auto read = [dtype](const Tensor& tensor, int i) {
                if (dtype == Dtype::kFloat32) return static_cast<const float*>(tensor.data_ptr())[i];
                const auto* bits = static_cast<const std::uint16_t*>(tensor.data_ptr()) + i;
                return dtype == Dtype::kFloat16 ? DecodeFloat16(bits) : DecodeBFloat16(bits);
            };
            require(grad.shape() == x.shape() && grad.dtype() == dtype && x.grad().device() == x.device(),
                    "explicit backward gradient metadata");
            for (int i = 0; i < 6; ++i) {
                require(read(grad, i) == 10 * weights[i], "explicit gradient shared branch accumulation");
                require(read(root_grad, i) == 2 * weights[i], "root explicit gradient retention");
                require(read(seed, i) == weights[i], "caller gradient must remain unchanged");
            }
        }
        invalid(output, Tensor());
        invalid(output, full({6}, 1, dtype, Device::kCUDA));
        invalid(output, full({2, 3}, 1, Dtype::kInt32, Device::kCUDA));
        if (output.device() == Device::kCUDA)
            invalid(output, full({2, 3}, 1, dtype, Device::kCPU));
        bool rejected = false;
        try { output.backward(); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "implicit gradient remains restricted to one element");
    }
    Tensor x = full({}, 3.0f, Dtype::kFloat32, Device::kCUDA);
    x.requires_grad_(true);
    Tensor output = mul(x, x);
    Tensor gradient = full({}, 2.0f, Dtype::kFloat32, Device::kCUDA);
    {
        NogradNode guard;
        output.backward(gradient);
        require(!EnableGrad(), "backward must preserve outer no-grad scope");
        Tensor grad = x.grad().to(Device::kCPU);
        require(*static_cast<const float*>(grad.data_ptr()) == 12, "explicit scalar gradient");
    }
    invalid(Tensor(), gradient);
    Tensor integer = full({}, 1, Dtype::kInt32, Device::kCUDA);
    invalid(integer, integer);
    bool rejected = false;
    try { x.backward(gradient); } catch (const std::logic_error&) { rejected = true; }
    require(rejected && EnableGrad(), "explicit backward still requires a graph node");
}
void TestSqueezeUnsqueeze() {
    auto require = [](bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    };
    for (Dtype dtype : {Dtype::kFloat32, Dtype::kFloat16, Dtype::kBFloat16}) {
        Tensor x = full({2, 3}, 2, dtype, Device::kCUDA);
        x.requires_grad_(true);
        Tensor view = x.transpose(0, 1);
        for (int dim = 0; dim <= view.dim(); ++dim) {
            Tensor expanded = view.unsqueeze(dim);
            auto expected_shape = view.shape().to_vector();
            expected_shape.insert(expected_shape.begin() + dim, 1);
            auto expected_stride = view.stride().to_vector();
            const std::size_t inserted = dim == view.dim() ? 1 : view.stride()[dim] * view.size(dim);
            expected_stride.insert(expected_stride.begin() + dim, inserted);
            require(expanded.shape() == expected_shape && expanded.stride() == expected_stride,
                    "unsqueeze must preserve strided mapping");
            require(expanded.data_ptr() == view.data_ptr() && expanded.grad_node() != view.grad_node(),
                    "unsqueeze storage and metadata");
            Tensor squeezed = expanded.squeeze(dim);
            require(squeezed.shape() == view.shape() && squeezed.stride() == view.stride(),
                    "squeeze must restore shape and stride");
            require(squeezed.data_ptr() == view.data_ptr(), "squeeze must share storage");
            require(view.squeeze(0).grad_node() == view.grad_node(), "no-op squeeze keeps graph");
            Tensor weights = full({3, 2}, 0, dtype, Device::kCUDA);
            {
                NogradNode guard;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 2; ++j) weights[i][j].fill_(i * 2 + j + 1);
            }
            mul(squeezed, weights).sum().backward();
        }
        {
            NogradNode guard;
            Tensor grad = x.grad().to(Device::kCPU);
            require(grad.shape() == x.shape() && grad.dtype() == dtype && x.grad().device() == x.device(),
                    "squeeze gradient metadata");
            const float expected[] = {3, 9, 15, 6, 12, 18};
            for (int i = 0; i < 6; ++i) {
                if (dtype == Dtype::kFloat32)
                    require(static_cast<const float*>(grad.data_ptr())[i] == expected[i], "squeeze float gradient");
                else {
                    const auto bits = static_cast<const std::uint16_t*>(grad.data_ptr())[i];
                    require(bits == (dtype == Dtype::kFloat16 ? EncodeFloat16(expected[i]) : EncodeBFloat16(expected[i])),
                            "squeeze half gradient");
                }
            }
            require(!x.unsqueeze(1).grad_node(), "no-grad unsqueeze");
            require(!x.unsqueeze(1).squeeze(1).grad_node(), "no-grad squeeze");
        }
    }
    Tensor scalar = full({}, 3.0f, Dtype::kFloat32, Device::kCUDA);
    scalar.requires_grad_(true);
    Tensor vector = scalar.unsqueeze(0);
    require(vector.shape() == std::vector<int>({1}) && !scalar.grad_node(), "scalar unsqueeze metadata");
    vector.squeeze(0).backward();
    {
        NogradNode guard;
        Tensor grad = scalar.grad().to(Device::kCPU);
        require(grad.is_scalar() && *static_cast<const float*>(grad.data_ptr()) == 1, "scalar squeeze gradient");
        Tensor row = full({2, 3}, 0.0f, Dtype::kFloat32, Device::kCUDA);
        row.unsqueeze(1)[1][0][2].fill_(7);
        Tensor host = row.to(Device::kCPU);
        require(static_cast<const float*>(host.data_ptr())[5] == 7, "unsqueeze shares writable storage");
    }
    Tensor integer = full({1, 2}, 1, Dtype::kInt32, Device::kCUDA);
    require(!integer.unsqueeze(0).grad_node() && !integer.squeeze(0).grad_node(), "integer layout no graph");
    for (int dim : {-1, 3}) {
        bool rejected = false;
        try { integer.squeeze(dim); } catch (const std::out_of_range&) { rejected = true; }
        require(rejected, "squeeze invalid dimension");
        rejected = false;
        try { integer.unsqueeze(dim); } catch (const std::out_of_range&) { rejected = true; }
        require(rejected, "unsqueeze invalid dimension");
    }
    Tensor empty;
    for (bool insert : {false, true}) {
        bool rejected = false;
        try { insert ? empty.unsqueeze(0) : empty.squeeze(0); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "empty layout input must be rejected");
    }
}
void TestIndexBackward() {
    auto require = [](bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    };
    for (Dtype dtype : {Dtype::kFloat32, Dtype::kFloat16, Dtype::kBFloat16}) {
        Tensor x = full({2, 2, 3}, 2, dtype, Device::kCUDA);
        x.requires_grad_(true);
        Tensor y = x[1];
        require(y.shape() == std::vector<int>({2, 3}), "index removes first dimension");
        require(y.grad_node() && !x.grad_node(), "index independent metadata");
        Tensor weights = full({3, 2}, 0, dtype, Device::kCUDA);
        {
            NogradNode guard;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 2; ++j) weights[i][j].fill_(i * 2 + j + 1);
        }
        mul(y.transpose(0, 1), weights).sum().backward();
        NogradNode guard;
        Tensor grad = x.grad().to(Device::kCPU);
        require(grad.shape() == x.shape() && grad.dtype() == dtype &&
                x.grad().device() == x.device(), "index gradient metadata");
        const float expected[] = {0, 0, 0, 0, 0, 0, 1, 3, 5, 2, 4, 6};
        for (int i = 0; i < 12; ++i) {
            if (dtype == Dtype::kFloat32)
                require(static_cast<const float*>(grad.data_ptr())[i] == expected[i], "index float gradient");
            else {
                const auto bits = static_cast<const std::uint16_t*>(grad.data_ptr())[i];
                require(bits == (dtype == Dtype::kFloat16 ? EncodeFloat16(expected[i]) : EncodeBFloat16(expected[i])),
                        "index half gradient");
            }
        }
        require(!x[1].grad_node(), "no-grad index must not record node");
    }
    Tensor x = full({2, 3}, 1.0f, Dtype::kFloat32, Device::kCUDA);
    x.requires_grad_(true);
    // A strided column selection and a chained scalar selection overlap.
    add(x.transpose(0, 1)[1].sum(), x[1][1]).backward();
    {
        NogradNode guard;
        Tensor grad = x.grad().to(Device::kCPU);
        const float expected[] = {0, 1, 0, 0, 2, 0};
        for (int i = 0; i < 6; ++i)
            require(static_cast<const float*>(grad.data_ptr())[i] == expected[i], "strided chained index gradient");
    }
    Tensor integer = full({2}, 1, Dtype::kInt32, Device::kCUDA);
    require(!integer[0].grad_node(), "integer index must not record node");
    for (int index : {-1, 2}) {
        bool rejected = false;
        try { integer[index]; } catch (const std::out_of_range&) { rejected = true; }
        require(rejected, "invalid index must be rejected");
    }
}
void TestNarrowBackward() {
    auto require = [](bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    };
    for (Dtype dtype : {Dtype::kFloat32, Dtype::kFloat16, Dtype::kBFloat16}) {
        Tensor x = full({2, 4}, 2, dtype, Device::kCUDA);
        x.requires_grad_(true);
        Tensor y = x.narrow(1, 1, 2);
        require(y.shape() == std::vector<int>({2, 2}) && !y.is_contiguous(), "narrow view layout");
        require(y.grad_node() && !x.grad_node(), "narrow independent metadata");
        Tensor weights = full({2, 2}, 0, dtype, Device::kCUDA);
        {
            NogradNode guard;
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) weights[i][j].fill_(i * 2 + j + 1);
        }
        // Transpose supplies a non-contiguous upstream gradient to NarrowNode.
        mul(y.transpose(0, 1), weights).sum().backward();
        {
            NogradNode guard;
            Tensor grad = x.grad().to(Device::kCPU);
            require(grad.shape() == x.shape() && grad.dtype() == dtype &&
                    x.grad().device() == x.device(), "narrow gradient metadata");
            const float expected[] = {0, 1, 3, 0, 0, 2, 4, 0};
            for (int i = 0; i < 8; ++i) {
                if (dtype == Dtype::kFloat32)
                    require(static_cast<const float*>(grad.data_ptr())[i] == expected[i], "narrow float gradient");
                else {
                    const auto bits = static_cast<const std::uint16_t*>(grad.data_ptr())[i];
                    require(bits == (dtype == Dtype::kFloat16 ? EncodeFloat16(expected[i]) : EncodeBFloat16(expected[i])),
                            "narrow half gradient");
                }
            }
            require(!x.narrow(1, 1, 2).grad_node(), "no-grad narrow must not record node");
        }
    }
    Tensor x = full({2, 4}, 1.0f, Dtype::kFloat32, Device::kCUDA);
    x.requires_grad_(true);
    // Non-contiguous input, chained narrow, and overlapping branches.
    Tensor view = x.transpose(0, 1);
    Tensor left = view.narrow(0, 0, 3).narrow(0, 1, 2);
    Tensor right = view.narrow(0, 2, 2);
    add(left.sum(), right.sum()).backward();
    {
        NogradNode guard;
        Tensor grad = x.grad().to(Device::kCPU);
        const float expected[] = {0, 1, 2, 1, 0, 1, 2, 1};
        for (int i = 0; i < 8; ++i)
            require(static_cast<const float*>(grad.data_ptr())[i] == expected[i], "chained overlapping narrow gradient");
    }
    Tensor integer = full({3}, 1, Dtype::kInt32, Device::kCUDA);
    require(!integer.narrow(0, 1, 1).grad_node(), "integer narrow must not record node");
}
void TestReshapeInference() {
    auto require = [](bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    };
    Tensor x = full({2, 3, 4}, 2.0f, Dtype::kFloat32, Device::kCUDA);
    x.requires_grad_(true);
    const std::vector<std::vector<int>> targets = {{-1, 4, 2}, {3, -1, 2}, {3, 2, -1}, {-1}};
    const std::vector<std::vector<int>> expected = {{3, 4, 2}, {3, 4, 2}, {3, 2, 4}, {24}};
    for (std::size_t i = 0; i < targets.size(); ++i) {
        Tensor y = x.reshape(targets[i]);
        require(y.shape() == expected[i] && y.data_ptr() == x.data_ptr(), "inferred reshape shape and storage");
        y.sum().backward();
    }
    {
        NogradNode guard;
        Tensor grad = x.grad().to(Device::kCPU);
        for (int i = 0; i < 24; ++i)
            require(static_cast<const float*>(grad.data_ptr())[i] == 4, "inferred reshape backward accumulation");
        require(!x.reshape({-1}).grad_node(), "no-grad inferred reshape");
    }
    Tensor scalar = full({}, 1.0f, Dtype::kFloat32, Device::kCUDA);
    require(scalar.reshape({-1}).shape() == std::vector<int>({1}), "scalar inferred reshape");
    const std::vector<std::vector<int>> invalid = {{-1, -1}, {-1, 5}, {0, -1}, {-2, 12}, {5, 5}};
    for (const auto& shape : invalid) {
        bool rejected = false;
        try { x.reshape(shape); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid inferred reshape must be rejected");
    }
    bool overflow = false;
    const int big = std::numeric_limits<int>::max();
    try { x.reshape({big, big, big}); }
    catch (const std::overflow_error&) { overflow = true; }
    require(overflow, "reshape shape multiplication overflow");
}
void TestReshapeBackward() {
    auto require = [](bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    };
    for (Dtype dtype : {Dtype::kFloat32, Dtype::kFloat16, Dtype::kBFloat16}) {
        Tensor x = full({6}, 2.0f, dtype, Device::kCUDA);
        x.requires_grad_(true);
        Tensor y = x.reshape({2, 3});
        require(y.data_ptr() == x.data_ptr(), "reshape must share storage");
        require(x.shape() == std::vector<int>({6}), "reshape must preserve input shape");
        require(y.grad_node() && !x.grad_node(), "reshape must use independent metadata");
        require(y.grad_node()->inputs[0].data_ptr() == x.data_ptr(), "reshape must save input");
        // Transpose backward produces a non-contiguous [2, 3] gradient.
        Tensor weights = full({3, 2}, 0.0f, dtype, Device::kCUDA);
        {
            NogradNode guard;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 2; ++j)
                    weights[i][j].fill_(i * 2 + j + 1);
        }
        Tensor loss = mul(y.transpose(0, 1), weights).sum();
        loss.backward();
        const float expected[] = {1, 3, 5, 2, 4, 6};
        {
            NogradNode guard;
            Tensor grad = x.grad().to(Device::kCPU);
            require(grad.shape() == x.shape() && grad.dtype() == dtype &&
                    x.grad().device() == x.device(), "reshape gradient metadata");
            for (int i = 0; i < 6; ++i) {
                float value;
                if (dtype == Dtype::kFloat32) value = static_cast<const float*>(grad.data_ptr())[i];
                else {
                    const auto* bits = static_cast<const std::uint16_t*>(grad.data_ptr()) + i;
                    value = dtype == Dtype::kFloat16 ? DecodeFloat16(bits) : DecodeBFloat16(bits);
                }
                require(value == expected[i], "non-contiguous reshape gradient value");
            }
            Tensor detached_view = x.reshape({3, 2});
            require(!detached_view.grad_node() && !detached_view.requires_grad(), "no-grad reshape metadata");
        }
        require(EnableGrad(), "reshape backward must restore grad recording");
        // sum backward supplies a zero-stride broadcast gradient.
        Tensor z = full({2, 3}, 1.0f, dtype, Device::kCUDA);
        z.requires_grad_(true);
        z.reshape({6}).sum().backward();
        require(z.grad().shape() == z.shape(), "broadcast reshape gradient shape");
        NogradNode guard;
        Tensor total = z.grad().sum().to(Device::kCPU);
        if (dtype == Dtype::kFloat32)
            require(*static_cast<const float*>(total.data_ptr()) == 6, "broadcast reshape gradient total");
        else {
            const auto bits = *static_cast<const std::uint16_t*>(total.data_ptr());
            require(bits == (dtype == Dtype::kFloat16 ? EncodeFloat16(6) : EncodeBFloat16(6)),
                    "half broadcast reshape gradient total");
        }
    }
    Tensor scalar = full({}, 3.0f, Dtype::kFloat32, Device::kCUDA);
    scalar.requires_grad_(true);
    scalar.reshape({1, 1}).reshape({}).backward();
    {
        NogradNode guard;
        Tensor grad = scalar.grad().to(Device::kCPU);
        require(grad.is_scalar() && *static_cast<const float*>(grad.data_ptr()) == 1,
                "scalar reshape chain gradient");
    }
    Tensor integer = full({2}, 1, Dtype::kInt32, Device::kCUDA);
    require(!integer.reshape({1, 2}).grad_node(), "integer reshape must not build graph");
    float values[] = {1, 2, 3, 4, 5, 6};
    Tensor matrix(values, {2, 3}, Dtype::kFloat32, Device::kCUDA);
    matrix.requires_grad_(true);
    Tensor transposed = matrix.transpose(0, 1);
    Tensor flat = transposed.reshape({-1});
    require(flat.is_contiguous() && flat.shape() == std::vector<int>({6}), "strided reshape layout");
    require(flat.data_ptr() != matrix.data_ptr(), "strided reshape must copy storage");
    float weight_values[] = {1, 2, 3, 4, 5, 6};
    Tensor weights(weight_values, {6}, Dtype::kFloat32, Device::kCUDA);
    mul(flat, weights).sum().backward();
    {
        NogradNode guard;
        Tensor host = flat.to(Device::kCPU);
        Tensor grad = matrix.grad().to(Device::kCPU);
        const float expected[] = {1, 4, 2, 5, 3, 6};
        const float expected_grad[] = {1, 3, 5, 2, 4, 6};
        for (int i = 0; i < 6; ++i) {
            require(static_cast<const float*>(host.data_ptr())[i] == expected[i], "strided reshape logical order");
            require(static_cast<const float*>(grad.data_ptr())[i] == expected_grad[i], "strided reshape backward");
        }
        flat.fill_(0);
        Tensor source = matrix.to(Device::kCPU);
        require(static_cast<const float*>(source.data_ptr())[0] == 1, "strided reshape storage independence");
        Tensor sliced = matrix.narrow(1, 1, 2).reshape({-1}).to(Device::kCPU);
        const float sliced_expected[] = {2, 3, 5, 6};
        for (int i = 0; i < 4; ++i)
            require(static_cast<const float*>(sliced.data_ptr())[i] == sliced_expected[i], "narrow reshape logical order");
        Tensor repeated = matrix.narrow(0, 0, 1).broadcast({2, 3}).reshape({-1}).to(Device::kCPU);
        for (int i = 0; i < 6; ++i)
            require(static_cast<const float*>(repeated.data_ptr())[i] == i % 3 + 1, "broadcast reshape logical order");
    }
}
void TestTransfer(Device source, Device destination) {
    float value = 3.0f;
    Tensor x(&value, {}, Dtype::kFloat32, source);
    x.requires_grad_(true);
    Tensor y = x.to(destination);
    Check(y.grad_node() != nullptr, "transfer must record a node");
    Check(y.to(destination).grad_node() == y.grad_node(), "same-device to must share metadata");
    Tensor loss = mul(y, y);
    loss.backward();
    CheckScalarGradient(x, 6.0f);
    Check(EnableGrad(), "backward must restore gradient recording");
}

void TestTransferChain() {
    float value = 2.0f;
    Tensor x(&value, {}, Dtype::kFloat32);
    x.requires_grad_(true);
    Tensor y = mul(x, x).to(Device::kCUDA).to(Device::kCPU);
    y.backward();
    CheckScalarGradient(x, 4.0f);
}

void TestNoGradAndInteger() {
    float value = 1.0f;
    Tensor x(&value, {}, Dtype::kFloat32);
    {
        NogradNode guard;
        Check(!x.to(Device::kCUDA).grad_node(), "no-grad transfer must not record a node");
    }
    int integer = 1;
    Tensor x_int(&integer, {}, Dtype::kInt32);
    Check(!x_int.to(Device::kCUDA).grad_node(), "integer transfer must not record a node");
}

void TestNonContiguousTransfer() {
    float values[] = {1, 2, 3, 4};
    Tensor x(values, {2, 2}, Dtype::kFloat32);
    Tensor view;
    {
        NogradNode guard;
        view = x.narrow(1, 0, 1);
    }
    view.requires_grad_(true);
    Tensor y = view.to(Device::kCUDA);
    auto transfer = y.grad_node();
    Check(dynamic_cast<ToDeviceNode*>(transfer.get()) != nullptr, "missing transfer node");
    Check(dynamic_cast<CloneNode*>(transfer->inputs[0].grad_node().get()) != nullptr,
          "non-contiguous transfer must retain clone node");
    Tensor seed(y.shape(), y.dtype(), y.device());
    seed.fill_(1);
    NogradNode guard;
    auto grads = transfer->backward(seed);
    Check(grads.size() == 1 && grads[0].device() == Device::kCPU &&
          grads[0].shape() == view.shape(), "non-contiguous transfer gradient metadata");
}
}  // namespace

int main() {
    try {
        int count = 0;
        cudaError_t status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && count == 0)) return 77;
        Check(status == cudaSuccess, "cudaGetDeviceCount failed");
        TestInplaceVersions();
        TestZeroGrad();
        TestExplicitBackwardGradient();
        TestSqueezeUnsqueeze();
        TestIndexBackward();
        TestNarrowBackward();
        TestReshapeInference();
        TestReshapeBackward();
        TestTransfer(Device::kCPU, Device::kCUDA);
        TestTransfer(Device::kCUDA, Device::kCPU);
        TestTransferChain();
        TestNoGradAndInteger();
        TestNonContiguousTransfer();
        Check(cudaDeviceSynchronize() == cudaSuccess, "CUDA synchronization failed");
        std::cout << "CUDA autograd tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

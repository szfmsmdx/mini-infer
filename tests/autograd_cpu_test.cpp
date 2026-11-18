#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mini_infer/autograd.h"
#include "mini_infer/ops/elementwise.h"
#include "mini_infer/tensor.h"

using namespace mtensor;

namespace mtensor {
struct TensorMetadataTestAccess {
    static void set_node(Tensor& tensor, std::shared_ptr<AutogradNode> node) {
        tensor.autograd_metadata_->grad_node = std::move(node);
    }
    static void set_grad(Tensor& tensor, const Tensor& grad) {
        tensor.autograd_metadata_->grad = grad;
    }
};
}

class TestNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor&) override { return {}; }
};

void CheckFloatValues(const Tensor& tensor, const std::vector<float>& expected) {
    if (tensor.numel() != expected.size()) {
        throw std::runtime_error("Unexpected gradient size");
    }
    const float* values = static_cast<const float*>(tensor.data_ptr());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (values[i] != expected[i]) {
            throw std::runtime_error("Unexpected gradient value");
        }
    }
}

Tensor MakeScalarTensor(float value) {
    Tensor tensor({}, Dtype::kFloat32);
    tensor.fill_(value);
    return tensor;
}

template <typename Exception, typename Operation>
void CheckThrows(Operation operation) {
    try {
        operation();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error("Expected backward exception");
}

void CheckBackwardEngine() {
    // Chain: unretained intermediates must still transmit gradients.
    {
        Tensor a = MakeScalarTensor(3);
        Tensor b = MakeScalarTensor(2);
        a.requires_grad_(true);
        b.requires_grad_(true);
        Tensor product = a * b;
        Tensor loss = neg(product);
        loss.backward();
        CheckFloatValues(a.grad(), {-2});
        CheckFloatValues(b.grad(), {-3});
        if (product.has_grad() || loss.has_grad()) {
            throw std::runtime_error("Unrequested gradients must not be retained");
        }
        if (a.grad().grad_node() || b.grad().grad_node() || !EnableGrad()) {
            throw std::runtime_error("Backward must not build a gradient graph");
        }
    }

    // Diamond with unequal branch depths: shared nodes execute once,
    // after all contributions arrive. Repeat without reusing old gradients.
    {
        Tensor a = MakeScalarTensor(3);
        a.requires_grad_(true);
        Tensor shared = a * a;
        shared.requires_grad_(true);
        Tensor loss = add(shared, neg(shared));
        loss.requires_grad_(true);
        loss.backward();
        CheckFloatValues(a.grad(), {0});
        CheckFloatValues(shared.grad(), {0});
        CheckFloatValues(loss.grad(), {1});

        Tensor square_loss = shared * shared;
        square_loss.requires_grad_(true);
        square_loss.backward();
        CheckFloatValues(a.grad(), {108});
        CheckFloatValues(shared.grad(), {18});
        square_loss.backward();
        CheckFloatValues(a.grad(), {216});
        CheckFloatValues(shared.grad(), {36});
        CheckFloatValues(square_loss.grad(), {2});
    }

    // Detach cuts the original graph but remains usable as a new leaf.
    {
        Tensor a = MakeScalarTensor(3);
        a.requires_grad_(true);
        Tensor intermediate = a * a;
        Tensor detached = intermediate.detach();
        detached.requires_grad_(true);
        Tensor loss = detached * a;
        loss.backward();
        CheckFloatValues(a.grad(), {9});
        CheckFloatValues(detached.grad(), {3});
        if (intermediate.has_grad()) {
            throw std::runtime_error("Detach must cut gradient propagation");
        }
    }

    // Existing graphs can run under an outer no-grad guard.
    {
        Tensor a = MakeScalarTensor(2);
        a.requires_grad_(true);
        Tensor loss = a * a;
        {
            NogradNode guard;
            loss.backward();
            if (EnableGrad()) throw std::runtime_error("Outer guard was lost");
            Tensor untracked = a * a;
            CheckThrows<std::logic_error>([&] { untracked.backward(); });
        }
        CheckFloatValues(a.grad(), {4});
        if (!EnableGrad()) throw std::runtime_error("Recording was not restored");
    }

    // All supported floating dtypes, including one-element non-scalar shape.
    for (Dtype dtype : {Dtype::kFloat32, Dtype::kFloat16, Dtype::kBFloat16}) {
        Tensor a({1}, dtype);
        a.fill_(2);
        a.requires_grad_(true);
        Tensor loss = a * a;
        loss.backward();
        Tensor expected({1}, dtype);
        expected.fill_(4);
        Tensor actual = a.grad();
        if (actual.dtype() != dtype || actual.shape() != a.shape() ||
            actual.device() != a.device()) {
            throw std::runtime_error("Gradient metadata mismatch");
        }
        const auto* bytes = static_cast<const unsigned char*>(actual.data_ptr());
        const auto* expected_bytes = static_cast<const unsigned char*>(expected.data_ptr());
        for (std::size_t i = 0; i < dtype_size(dtype); ++i) {
            if (bytes[i] != expected_bytes[i]) {
                throw std::runtime_error("Floating dtype backward mismatch");
            }
        }
    }

    // A composite expression covers all five operators; compare both inputs
    // against independent central finite differences.
    {
        Tensor a = MakeScalarTensor(1.5f);
        Tensor b = MakeScalarTensor(2.0f);
        a.requires_grad_(true);
        b.requires_grad_(true);
        Tensor loss = neg(div(sub(mul(a, b), a), add(a, b)));
        loss.backward();
        auto evaluate = [](float x, float y) {
            return -(x * y - x) / (x + y);
        };
        const float epsilon = 0.001f;
        float da = (evaluate(1.5f + epsilon, 2) - evaluate(1.5f - epsilon, 2)) /
                   (2 * epsilon);
        float db = (evaluate(1.5f, 2 + epsilon) - evaluate(1.5f, 2 - epsilon)) /
                   (2 * epsilon);
        float actual_a = *static_cast<const float*>(a.grad().data_ptr());
        float actual_b = *static_cast<const float*>(b.grad().data_ptr());
        if (std::fabs(actual_a - da) > 0.0002f ||
            std::fabs(actual_b - db) > 0.0002f) {
            throw std::runtime_error("Backward disagrees with finite differences");
        }
    }

    // Invalid roots and malformed Node output; guard restores after errors.
    {
        Tensor empty;
        CheckThrows<std::invalid_argument>([&] { empty.backward(); });
        Tensor vector({2}, Dtype::kFloat32);
        CheckThrows<std::invalid_argument>([&] { vector.backward(); });
        Tensor leaf = MakeScalarTensor(1);
        CheckThrows<std::logic_error>([&] { leaf.backward(); });
        for (Dtype dtype : {Dtype::kInt8, Dtype::kUInt8, Dtype::kInt16,
                            Dtype::kInt32, Dtype::kInt64}) {
            Tensor integer({}, dtype);
            CheckThrows<std::invalid_argument>([&] { integer.backward(); });
        }
        Tensor malformed = MakeScalarTensor(1);
        auto node = std::make_shared<TestNode>();
        node->inputs = {leaf};
        malformed.set_grad_node(node);
        CheckThrows<std::logic_error>([&] { malformed.backward(); });
        if (!EnableGrad()) throw std::runtime_error("Exception failed to restore recording");
    }
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
    Tensor base = full({2, 3}, 3.0f, Dtype::kFloat32, Device::kCPU);
    std::vector<Tensor> aliases = {
        base, base[0], base.narrow(1, 1, 2), base.transpose(0, 1), base.permute({1, 0}),
        base.reshape({6}), base.unsqueeze(1), base.unsqueeze(1).squeeze(1), base.detach(),
        base.to(Device::kCPU), base.narrow(0, 0, 1).broadcast({2, 3})
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
    try { base.copy_(full({1}, 0.0f, Dtype::kFloat32, Device::kCPU)); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && base.version() == version, "failed copy validation must not update version");
    rejected = false;
    try { aliases.back().fill_(1); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && base.version() == version, "rejected broadcast write must not update version");
    Tensor scalar = full({}, 2.0f, Dtype::kFloat32, Device::kCPU);
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
                    Tensor a = full({2, 3}, 6, dtype, Device::kCPU);
                    Tensor b = full({3}, 2, dtype, Device::kCPU);
                    a.requires_grad_(true);
                    // b does not request a gradient, but its value is still needed for a's gradient.
                    Tensor loss = operation(a, b).sum();
                    Tensor target = side == 0 ? a : b;
                    auto before = target.version();
                    {
                        NogradNode guard;
                        if (write == 0) target.fill_(4);
                        else if (write == 1) target.zero_();
                        else if (write == 2) target.copy_(full(target.shape(), 4, dtype, Device::kCPU));
                        else target.detach().narrow(0, 0, 1).fill_(4);
                    }
                    require(target.version() == before + 1, "in-place write must update shared version once");
                    rejects_modified(loss);
                }
            }
        }
    }
    Tensor x = full({2, 3}, 3.0f, Dtype::kFloat32, Device::kCPU);
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
            const auto bits = static_cast<const std::uint16_t*>(tensor.data_ptr())[i];
            return dtype == Dtype::kFloat16 ? static_cast<float>(Float16::from_bits(bits))
                                            : static_cast<float>(BFloat16::from_bits(bits));
        };
        auto check_values = [&require, &read](const Tensor& tensor, float expected) {
            NogradNode guard;
            Tensor host = tensor.to(Device::kCPU);
            for (std::size_t i = 0; i < host.numel(); ++i)
                require(read(host, i) == expected, "zero_grad gradient value");
        };
        Tensor x = full({2}, 3, dtype, Device::kCPU);
        x.requires_grad_(true);
        x.zero_grad();
        require(!x.has_grad() && x.requires_grad(), "zero_grad on fresh leaf must be a no-op");
        Tensor output = mul(x, x);
        output.requires_grad_(true);
        auto node = output.grad_node();
        Tensor gradient = full({2}, 1, dtype, Device::kCPU);
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
    Tensor source = full({}, 1.0f, Dtype::kFloat32, Device::kCPU);
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
        Tensor x = full({2, 3}, 2, dtype, Device::kCPU);
        x.requires_grad_(true);
        Tensor output = add(mul(x, x), x);
        output.requires_grad_(true);
        Tensor values = full({3, 2}, 0, dtype, Device::kCPU);
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
                const auto bits = static_cast<const std::uint16_t*>(tensor.data_ptr())[i];
                return dtype == Dtype::kFloat16 ? static_cast<float>(Float16::from_bits(bits))
                                                : static_cast<float>(BFloat16::from_bits(bits));
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
        invalid(output, full({6}, 1, dtype, Device::kCPU));
        invalid(output, full({2, 3}, 1, Dtype::kInt32, Device::kCPU));
        if (output.device() == Device::kCUDA)
            invalid(output, full({2, 3}, 1, dtype, Device::kCPU));
        bool rejected = false;
        try { output.backward(); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "implicit gradient remains restricted to one element");
    }
    Tensor x = full({}, 3.0f, Dtype::kFloat32, Device::kCPU);
    x.requires_grad_(true);
    Tensor output = mul(x, x);
    Tensor gradient = full({}, 2.0f, Dtype::kFloat32, Device::kCPU);
    {
        NogradNode guard;
        output.backward(gradient);
        require(!EnableGrad(), "backward must preserve outer no-grad scope");
        Tensor grad = x.grad().to(Device::kCPU);
        require(*static_cast<const float*>(grad.data_ptr()) == 12, "explicit scalar gradient");
    }
    invalid(Tensor(), gradient);
    Tensor integer = full({}, 1, Dtype::kInt32, Device::kCPU);
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
        Tensor x = full({2, 3}, 2, dtype, Device::kCPU);
        x.requires_grad_(true);
        Tensor view = x.transpose(0, 1);
        for (int dim = 0; dim <= view.dim(); ++dim) {
            Tensor expanded = view.unsqueeze(dim);
            auto expected_shape = view.shape();
            expected_shape.insert(expected_shape.begin() + dim, 1);
            auto expected_stride = view.stride();
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
            Tensor weights = full({3, 2}, 0, dtype, Device::kCPU);
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
    Tensor scalar = full({}, 3.0f, Dtype::kFloat32, Device::kCPU);
    scalar.requires_grad_(true);
    Tensor vector = scalar.unsqueeze(0);
    require(vector.shape() == std::vector<int>({1}) && !scalar.grad_node(), "scalar unsqueeze metadata");
    vector.squeeze(0).backward();
    {
        NogradNode guard;
        Tensor grad = scalar.grad().to(Device::kCPU);
        require(grad.is_scalar() && *static_cast<const float*>(grad.data_ptr()) == 1, "scalar squeeze gradient");
        Tensor row = full({2, 3}, 0.0f, Dtype::kFloat32, Device::kCPU);
        row.unsqueeze(1)[1][0][2].fill_(7);
        Tensor host = row.to(Device::kCPU);
        require(static_cast<const float*>(host.data_ptr())[5] == 7, "unsqueeze shares writable storage");
    }
    Tensor integer = full({1, 2}, 1, Dtype::kInt32, Device::kCPU);
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
        Tensor x = full({2, 2, 3}, 2, dtype, Device::kCPU);
        x.requires_grad_(true);
        Tensor y = x[1];
        require(y.shape() == std::vector<int>({2, 3}), "index removes first dimension");
        require(y.grad_node() && !x.grad_node(), "index independent metadata");
        Tensor weights = full({3, 2}, 0, dtype, Device::kCPU);
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
    Tensor x = full({2, 3}, 1.0f, Dtype::kFloat32, Device::kCPU);
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
    Tensor integer = full({2}, 1, Dtype::kInt32, Device::kCPU);
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
        Tensor x = full({2, 4}, 2, dtype, Device::kCPU);
        x.requires_grad_(true);
        Tensor y = x.narrow(1, 1, 2);
        require(y.shape() == std::vector<int>({2, 2}) && !y.is_contiguous(), "narrow view layout");
        require(y.grad_node() && !x.grad_node(), "narrow independent metadata");
        Tensor weights = full({2, 2}, 0, dtype, Device::kCPU);
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
    Tensor x = full({2, 4}, 1.0f, Dtype::kFloat32, Device::kCPU);
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
    Tensor integer = full({3}, 1, Dtype::kInt32, Device::kCPU);
    require(!integer.narrow(0, 1, 1).grad_node(), "integer narrow must not record node");
}
void TestReshapeInference() {
    auto require = [](bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    };
    Tensor x = full({2, 3, 4}, 2.0f, Dtype::kFloat32, Device::kCPU);
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
    Tensor scalar = full({}, 1.0f, Dtype::kFloat32, Device::kCPU);
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
        Tensor x = full({6}, 2.0f, dtype, Device::kCPU);
        x.requires_grad_(true);
        Tensor y = x.reshape({2, 3});
        require(y.data_ptr() == x.data_ptr(), "reshape must share storage");
        require(x.shape() == std::vector<int>({6}), "reshape must preserve input shape");
        require(y.grad_node() && !x.grad_node(), "reshape must use independent metadata");
        require(y.grad_node()->inputs[0].data_ptr() == x.data_ptr(), "reshape must save input");
        // Transpose backward produces a non-contiguous [2, 3] gradient.
        Tensor weights = full({3, 2}, 0.0f, dtype, Device::kCPU);
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
                    const auto bits = static_cast<const std::uint16_t*>(grad.data_ptr())[i];
                    value = dtype == Dtype::kFloat16 ? static_cast<float>(Float16::from_bits(bits))
                                                    : static_cast<float>(BFloat16::from_bits(bits));
                }
                require(value == expected[i], "non-contiguous reshape gradient value");
            }
            Tensor detached_view = x.reshape({3, 2});
            require(!detached_view.grad_node() && !detached_view.requires_grad(), "no-grad reshape metadata");
        }
        require(EnableGrad(), "reshape backward must restore grad recording");
        // sum backward supplies a zero-stride broadcast gradient.
        Tensor z = full({2, 3}, 1.0f, dtype, Device::kCPU);
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
    Tensor scalar = full({}, 3.0f, Dtype::kFloat32, Device::kCPU);
    scalar.requires_grad_(true);
    scalar.reshape({1, 1}).reshape({}).backward();
    {
        NogradNode guard;
        Tensor grad = scalar.grad().to(Device::kCPU);
        require(grad.is_scalar() && *static_cast<const float*>(grad.data_ptr()) == 1,
                "scalar reshape chain gradient");
    }
    Tensor integer = full({2}, 1, Dtype::kInt32, Device::kCPU);
    require(!integer.reshape({1, 2}).grad_node(), "integer reshape must not build graph");
    float values[] = {1, 2, 3, 4, 5, 6};
    Tensor matrix(values, {2, 3}, Dtype::kFloat32, Device::kCPU);
    matrix.requires_grad_(true);
    Tensor transposed = matrix.transpose(0, 1);
    Tensor flat = transposed.reshape({-1});
    require(flat.is_contiguous() && flat.shape() == std::vector<int>({6}), "strided reshape layout");
    require(flat.data_ptr() != matrix.data_ptr(), "strided reshape must copy storage");
    float weight_values[] = {1, 2, 3, 4, 5, 6};
    Tensor weights(weight_values, {6}, Dtype::kFloat32, Device::kCPU);
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
int main() {
    TestInplaceVersions();
    TestZeroGrad();
    TestExplicitBackwardGradient();
    TestSqueezeUnsqueeze();
    TestIndexBackward();
    TestNarrowBackward();
    TestReshapeInference();
    TestReshapeBackward();
    CheckBackwardEngine();
    {
        Tensor x = MakeScalarTensor(3);
        x.requires_grad_(true);
        Tensor y = x.clone();
        Tensor loss = mul(y, y);
        loss.backward();
        CheckFloatValues(x.grad(), {6.0f});
    }
    if (!EnableGrad()) throw std::runtime_error("gradient recording must be enabled initially");
    {
        NogradNode outer;
        if (EnableGrad()) throw std::runtime_error("guard must disable gradient recording");
        {
            NogradNode inner;
            if (EnableGrad()) throw std::runtime_error("nested guard must keep recording disabled");
        }
        if (EnableGrad()) throw std::runtime_error("nested guard must restore disabled state");
    }
    if (!EnableGrad()) throw std::runtime_error("guard must restore enabled state");
    try {
        NogradNode guard;
        throw std::runtime_error("test exception");
    } catch (const std::runtime_error&) {
        if (!EnableGrad()) throw std::runtime_error("guard must restore state after exception");
    }

    for (Dtype dtype : {Dtype::kFloat32, Dtype::kFloat16, Dtype::kBFloat16}) {
        Tensor x({1}, dtype);
        Tensor copy = x;
        if (x.has_grad() || !x.grad().empty()) {
            throw std::runtime_error("missing gradient must return an empty tensor");
        }
        Tensor saved_grad({1}, dtype);
        TensorMetadataTestAccess::set_grad(x, saved_grad);
        if (!copy.has_grad() || copy.grad().data_ptr() != saved_grad.data_ptr()) {
            throw std::runtime_error("gradient must be shared and readable through copies");
        }
        if (x.requires_grad() || &x.requires_grad_(true) != &x || !copy.requires_grad()) {
            throw std::runtime_error("gradient state must be shared by copies");
        }
        copy.requires_grad_(false);
        if (x.requires_grad()) throw std::runtime_error("gradient state was not reset");
        TensorMetadataTestAccess::set_node(x, std::make_shared<TestNode>());
        x.requires_grad_(true);
        if (!copy.requires_grad()) throw std::runtime_error("non-leaf gradient state must be editable");
        x.requires_grad_(false);
        Tensor moved = std::move(x);
        if (x.requires_grad()) throw std::runtime_error("moved-from state must be empty");
        if (x.has_grad() || !x.grad().empty() || !moved.has_grad()) {
            throw std::runtime_error("gradient must move with metadata");
        }
        x.requires_grad_(true);
        if (!x.requires_grad() || moved.requires_grad()) {
            throw std::runtime_error("moved-from tensor must get independent metadata");
        }
    }
    for (Dtype dtype : {Dtype::kInt8, Dtype::kUInt8, Dtype::kInt16, Dtype::kInt32, Dtype::kInt64}) {
        Tensor x({1}, dtype);
        x.requires_grad_(true);
        if (!x.requires_grad()) throw std::runtime_error("gradient flag must be settable for integer tensors");
    }

    // grad_node / set_grad_node
    {
        Tensor x({2}, Dtype::kFloat32);
        if (x.grad_node()) throw std::runtime_error("new tensor must not have a graph node");
        auto node = std::make_shared<TestNode>();
        x.set_grad_node(node);
        if (x.grad_node() != node) throw std::runtime_error("graph node was not stored");
        if (x.requires_grad()) throw std::runtime_error("set_grad_node must not change requires_grad");

        Tensor copy = x;
        if (copy.grad_node() != node) throw std::runtime_error("graph node must be shared by copies");
        x.set_grad_node(nullptr);
        if (copy.grad_node()) throw std::runtime_error("cleared graph node must be visible through copies");
    }

    // accumulate_grad: first write, independence from source, then addition
    {
        float initial[] = {0.0f, 0.0f};
        float first[] = {1.0f, 2.0f};
        float second[] = {3.0f, 4.0f};
        Tensor x(initial, {2}, Dtype::kFloat32);
        Tensor g1(first, {2}, Dtype::kFloat32);
        x.accumulate_grad(g1);
        if (!x.has_grad()) throw std::runtime_error("first gradient was not stored");
        CheckFloatValues(x.grad(), {1.0f, 2.0f});

        g1.fill_(99.0f);
        CheckFloatValues(x.grad(), {1.0f, 2.0f});

        Tensor copy = x;
        Tensor g2(second, {2}, Dtype::kFloat32);
        copy.accumulate_grad(g2);
        CheckFloatValues(x.grad(), {4.0f, 6.0f});
    }

    // detach: shared storage, independent autograd state
    {
        float values[] = {5.0f, 6.0f};
        float grad_values[] = {1.0f, 1.0f};
        Tensor x(values, {2}, Dtype::kFloat32);
        x.requires_grad_(true);
        auto node = std::make_shared<TestNode>();
        x.set_grad_node(node);
        Tensor g(grad_values, {2}, Dtype::kFloat32);
        x.accumulate_grad(g);

        Tensor detached = x.detach();
        if (detached.data_ptr() != x.data_ptr()) throw std::runtime_error("detach must share storage");
        if (detached.shape() != x.shape() || detached.stride() != x.stride()) {
            throw std::runtime_error("detach must preserve metadata");
        }
        if (detached.requires_grad() || detached.grad_node() || detached.has_grad()) {
            throw std::runtime_error("detach must use fresh autograd metadata");
        }

        static_cast<float*>(detached.data_ptr())[0] = 7.0f;
        const float* x_values = static_cast<const float*>(x.data_ptr());
        if (x_values[0] != 7.0f) throw std::runtime_error("detach must share data");
        if (!x.requires_grad() || x.grad_node() != node || !x.has_grad()) {
            throw std::runtime_error("detach must not change source autograd state");
        }
    }

    // add: float dtypes build a graph, integer dtypes and no-grad mode do not
    {
        float left[] = {1.0f, 2.0f};
        float right[] = {3.0f, 4.0f};
        float seed_values[] = {5.0f, 6.0f};
        Tensor a(left, {2}, Dtype::kFloat32);
        Tensor b(right, {2}, Dtype::kFloat32);
        Tensor out = add(a, b);
        auto node = out.grad_node();
        if (!node) throw std::runtime_error("float add must create a graph node");
        if (out.requires_grad()) throw std::runtime_error("graph construction must not change requires_grad");

        Tensor seed(seed_values, {2}, Dtype::kFloat32);
        std::vector<Tensor> grads = node->backward(seed);
        if (grads.size() != 2) throw std::runtime_error("AddNode must return two gradients");
        CheckFloatValues(grads[0], {5.0f, 6.0f});
        CheckFloatValues(grads[1], {5.0f, 6.0f});

        for (Dtype dtype : {Dtype::kFloat16, Dtype::kBFloat16}) {
            Tensor x({1}, dtype);
            Tensor y({1}, dtype);
            x.fill_(1);
            y.fill_(1);
            if (!add(x, y).grad_node()) throw std::runtime_error("16-bit float add must build a graph");
        }
    }
    {
        std::int32_t left[] = {1, 2};
        std::int32_t right[] = {3, 4};
        Tensor a(left, {2}, Dtype::kInt32);
        Tensor b(right, {2}, Dtype::kInt32);
        if (add(a, b).grad_node()) throw std::runtime_error("integer add must not build a graph");
    }
    {
        float left[] = {1.0f};
        float right[] = {2.0f};
        Tensor a(left, {1}, Dtype::kFloat32);
        Tensor b(right, {1}, Dtype::kFloat32);
        NogradNode guard;
        if (add(a, b).grad_node()) throw std::runtime_error("NogradNode must disable graph construction");
    }

    // sub / neg / mul / div backward formulas
    {
        float a_values[] = {6.0f, 4.0f};
        float b_values[] = {2.0f, 2.0f};
        float g_values[] = {4.0f, 6.0f};
        Tensor a(a_values, {2}, Dtype::kFloat32);
        Tensor b(b_values, {2}, Dtype::kFloat32);
        Tensor g(g_values, {2}, Dtype::kFloat32);

        Tensor sub_out = sub(a, b);
        Tensor neg_out = neg(a);
        Tensor mul_out = mul(a, b);
        Tensor div_out = div(a, b);

        NogradNode guard;

        std::vector<Tensor> grads = sub_out.grad_node()->backward(g);
        if (grads.size() != 2) throw std::runtime_error("SubNode must return two gradients");
        CheckFloatValues(grads[0], {4.0f, 6.0f});
        CheckFloatValues(grads[1], {-4.0f, -6.0f});

        grads = neg_out.grad_node()->backward(g);
        if (grads.size() != 1) throw std::runtime_error("NegNode must return one gradient");
        CheckFloatValues(grads[0], {-4.0f, -6.0f});

        grads = mul_out.grad_node()->backward(g);
        if (grads.size() != 2) throw std::runtime_error("MulNode must return two gradients");
        CheckFloatValues(grads[0], {8.0f, 12.0f});
        CheckFloatValues(grads[1], {24.0f, 24.0f});

        grads = div_out.grad_node()->backward(g);
        if (grads.size() != 2) throw std::runtime_error("DivNode must return two gradients");
        CheckFloatValues(grads[0], {2.0f, 3.0f});
        CheckFloatValues(grads[1], {-6.0f, -6.0f});
    }

    // Views retain the input graph and use independent metadata.
    {
        float a_values[] = {1.0f, 2.0f};
        float b_values[] = {3.0f, 4.0f};
        Tensor a(a_values, {2}, Dtype::kFloat32);
        Tensor b(b_values, {2}, Dtype::kFloat32);
        Tensor out = add(a, b);

        Tensor indexed = out[0];
        if (!indexed.grad_node() || indexed.grad_node()->inputs[0].grad_node() != out.grad_node()) {
            throw std::runtime_error("index must retain the input graph");
        }
        Tensor narrowed = out.narrow(0, 0, 1);
        if (!narrowed.grad_node() || narrowed.grad_node()->inputs[0].grad_node() != out.grad_node()) {
            throw std::runtime_error("narrow must retain the input graph");
        }
        Tensor reshaped = out.reshape({1, 2});
        if (!reshaped.grad_node() || reshaped.grad_node()->inputs[0].grad_node() != out.grad_node()) {
            throw std::runtime_error("reshape must retain the input graph");
        }

        Tensor leaf(a_values, {2}, Dtype::kFloat32);
        leaf.requires_grad_(true);
        if (!leaf[0].grad_node() || leaf.grad_node()) {
            throw std::runtime_error("index must record a node without modifying the leaf metadata");
        }

        Tensor same_device = leaf.to(Device::kCPU);
        if (same_device.data_ptr() != leaf.data_ptr()) {
            throw std::runtime_error("same-device to must share storage");
        }

        NogradNode guard;
        Tensor view = out[0];
        Tensor copy = out.clone();
        if (view.data_ptr() != out.data_ptr()) {
            throw std::runtime_error("no-grad mode must allow views");
        }
        if (copy.data_ptr() == out.data_ptr()) {
            throw std::runtime_error("no-grad clone must copy storage");
        }
    }
}

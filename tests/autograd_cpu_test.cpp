#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "autograd/autograd.h"
#include "ops/elementwise.h"
#include "tensor/tensor.h"

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

template <typename Operation>
void CheckAutogradProtection(Operation operation, const char* message) {
    try {
        operation();
    } catch (const std::logic_error& error) {
        if (std::string(error.what()).find("autograd") != std::string::npos) {
            return;
        }
        throw std::runtime_error(message);
    }
    throw std::runtime_error(message);
}

Tensor Scalar(float value) {
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
        Tensor a = Scalar(3);
        Tensor b = Scalar(2);
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
        Tensor a = Scalar(3);
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
        Tensor a = Scalar(3);
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
        Tensor a = Scalar(2);
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
        Tensor a = Scalar(1.5f);
        Tensor b = Scalar(2.0f);
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
        Tensor leaf = Scalar(1);
        CheckThrows<std::logic_error>([&] { leaf.backward(); });
        for (Dtype dtype : {Dtype::kInt8, Dtype::kUInt8, Dtype::kInt16,
                            Dtype::kInt32, Dtype::kInt64}) {
            Tensor integer({}, dtype);
            CheckThrows<std::invalid_argument>([&] { integer.backward(); });
        }
        Tensor malformed = Scalar(1);
        auto node = std::make_shared<TestNode>();
        node->inputs = {leaf};
        malformed.set_grad_node(node);
        CheckThrows<std::logic_error>([&] { malformed.backward(); });
        if (!EnableGrad()) throw std::runtime_error("Exception failed to restore recording");
    }
}

int main() {
    CheckBackwardEngine();
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

    // Autograd protection for clone / cross-device to / views
    {
        float a_values[] = {1.0f, 2.0f};
        float b_values[] = {3.0f, 4.0f};
        Tensor a(a_values, {2}, Dtype::kFloat32);
        Tensor b(b_values, {2}, Dtype::kFloat32);
        Tensor out = add(a, b);

        CheckAutogradProtection([&] { out.clone(); }, "clone on graph tensor must be rejected");
        CheckAutogradProtection([&] { out.to(Device::kCUDA); }, "cross-device to on graph tensor must be rejected");
        CheckAutogradProtection([&] { out[0]; }, "operator[] on graph tensor must be rejected");
        CheckAutogradProtection([&] { out.narrow(0, 0, 1); }, "narrow on graph tensor must be rejected");
        CheckAutogradProtection([&] { out.reshape({1, 2}); }, "reshape on graph tensor must be rejected");

        Tensor leaf(a_values, {2}, Dtype::kFloat32);
        leaf.requires_grad_(true);
        CheckAutogradProtection([&] { leaf.clone(); }, "clone on leaf requiring grad must be rejected");
        CheckAutogradProtection([&] { leaf[0]; }, "operator[] on leaf requiring grad must be rejected");

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

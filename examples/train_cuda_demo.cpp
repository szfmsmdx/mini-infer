#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>

#include "device/memory.h"
#include "mini_infer/autograd.h"
#include "mini_infer/modules/linear.h"
#include "mini_infer/optim/adam.h"
#include "utils/logger.h"

namespace {

float ReadScalar(const mtensor::Tensor& tensor) {
    mtensor::NogradNode guard;
    const mtensor::Tensor host = tensor.to(mtensor::Device::kCPU);
    return *static_cast<const float*>(host.data_ptr());
}

void LogInfo(mtensor::Logger& training_log, const std::string& message) {
    training_log.info(message);
    std::cout << message << std::endl;
}

}  // namespace

int main() {
    using namespace mtensor;

    try {
        constexpr Device device = Device::kCUDA;
        constexpr Dtype dtype = Dtype::kFloat32;
        constexpr int steps = 250;

        Logger training_log("train_cuda_demo.log");
        int device_id = 0;
        cudaDeviceProp device_info{};
        CUDA_CHECK(cudaGetDevice(&device_id));
        CUDA_CHECK(cudaGetDeviceProperties(&device_info, device_id));
        LogInfo(training_log, std::string("CUDA FP32 linear regression with Adam on ") + device_info.name);

        // 四条训练样本，标签对应 y = 2*x1 - 3*x2 + 0.5。
        const float inputs[] = {-1, -1, -1, 1, 1, -1, 1, 1};
        const float labels[] = {1.5f, -4.5f, 5.5f, -0.5f};
        const Tensor x(inputs, {4, 2}, dtype, device);
        const Tensor target(labels, {4, 1}, dtype, device);
        Tensor weight = full({1, 2}, 0.0f, dtype, device);
        Tensor bias = full({1}, 0.0f, dtype, device);
        weight.requires_grad_(true);
        bias.requires_grad_(true);

        Linear model(weight, bias);
        Adam optimizer({weight, bias}, 0.1f);

        double total_ms = 0.0;
        for (int step = 1; step <= steps; ++step) {
            const auto start = std::chrono::steady_clock::now();
            optimizer.zero_grad();
            const Tensor error = model(x) - target;
            Tensor loss = (error * error).mean();
            loss.backward();
            optimizer.step();
            // 等 GPU 完成参数更新，再计算本步耗时。
            CUDA_CHECK(cudaDeviceSynchronize());
            const double elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            total_ms += elapsed_ms;

            std::ostringstream message;
            message << "step=" << step << '/' << steps << " loss=" << ReadScalar(loss)
                    << " elapsed_ms=" << std::fixed << std::setprecision(3) << elapsed_ms;
            LogInfo(training_log, message.str());
        }

        NogradNode guard;
        const Tensor error = model(x) - target;
        const float final_loss = ReadScalar((error * error).mean());
        const Tensor host_weight = weight.to(Device::kCPU);
        const auto* values = static_cast<const float*>(host_weight.data_ptr());

        const float test_input[] = {2, -1};
        const Tensor test_x(test_input, {1, 2}, dtype, device);
        const float prediction = ReadScalar(model(test_x));
        std::ostringstream summary;
        summary << "final_loss=" << final_loss << " weight=[" << values[0] << ", "
                << values[1] << "] bias=" << ReadScalar(bias)
                << " prediction_for_[2,-1]=" << prediction << " (expected 7.5)"
                << " mean_step_ms=" << total_ms / steps;
        LogInfo(training_log, summary.str());
        if (!(final_loss < 1e-6f && std::abs(prediction - 7.5f) < 1e-3f)) {
            training_log.error("Training did not converge.");
            std::cerr << "Training did not converge.\n";
            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr << "CUDA training demo failed: " << error.what() << '\n';
        return 1;
    }
}

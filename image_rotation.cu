// ============================================================================
// High-Throughput GPU Image Rotation Routine
// ============================================================================
// Based on the CUDA C++ lecture example:
// "Introduction to CUDA Programming.cu"
//
// Build on Linux / Google Colab / HPC Cluster with NVIDIA CUDA Toolkit:
//   nvcc -O3 -std=c++14 image_rotation.cu -o image_rotation
//
// Run the compiled executable:
//   ./image_rotation
// ============================================================================

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Type aliases aligned with lecture convention
using Image = std::vector<unsigned char>;
using Clock = std::chrono::steady_clock;

// CUDA error-checking helper function (lecture pattern)
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// Elapsed time helper in seconds
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// RAII GPU Buffer Wrapper from lecture material
struct DeviceBuffer {
    unsigned char* data = nullptr;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(std::size_t bytes) {
        check(cudaMalloc(reinterpret_cast<void**>(&data), bytes), "Allocate GPU memory");
    }

    ~DeviceBuffer() {
        if (data) {
            cudaFree(data);
        }
    }
};

// ----------------------------------------------------------------------------
// Helper: Synthetic RGB Image Generation & PPM File Export
// ----------------------------------------------------------------------------
Image make_image(int size, std::mt19937& rng) {
    Image image(static_cast<std::size_t>(size) * size * 3);
    std::uniform_int_distribution<int> pixel(0, 255);
    for (auto& value : image) {
        value = static_cast<unsigned char>(pixel(rng));
    }
    return image;
}

void save_ppm(const char* filename, const Image& image, int size) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error(std::string("Cannot create file ") + filename);
    }
    file << "P6\n" << size << ' ' << size << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.data()),
               static_cast<std::streamsize>(image.size()));
    if (!file) {
        throw std::runtime_error(std::string("Cannot write to file ") + filename);
    }
}

// ----------------------------------------------------------------------------
// Part A: CPU Baseline Reference Rotation Routine
// ----------------------------------------------------------------------------
Image rotate_cpu(const Image& input, int size, float angle_deg) {
    Image output(input.size(), 0);
    const float rad = angle_deg * (3.14159265358979323846f / 180.0f);
    const float cos_a = std::cos(rad);
    const float sin_a = std::sin(rad);
    const float center = (static_cast<float>(size) - 1.0f) / 2.0f;

    for (int y_dst = 0; y_dst < size; ++y_dst) {
        for (int x_dst = 0; x_dst < size; ++x_dst) {
            float x_shift = static_cast<float>(x_dst) - center;
            float y_shift = static_cast<float>(y_dst) - center;

            // Inverse rotation linear algebra formula
            int src_x = static_cast<int>(std::round(x_shift * cos_a + y_shift * sin_a + center));
            int src_y = static_cast<int>(std::round(-x_shift * sin_a + y_shift * cos_a + center));

            std::size_t dst_offset = (static_cast<std::size_t>(y_dst) * size + x_dst) * 3;

            if (src_x >= 0 && src_x < size && src_y >= 0 && src_y < size) {
                std::size_t src_offset = (static_cast<std::size_t>(src_y) * size + src_x) * 3;
                output[dst_offset]     = input[src_offset];
                output[dst_offset + 1] = input[src_offset + 1];
                output[dst_offset + 2] = input[src_offset + 2];
            } else {
                output[dst_offset]     = 0;
                output[dst_offset + 1] = 0;
                output[dst_offset + 2] = 0;
            }
        }
    }
    return output;
}

// ----------------------------------------------------------------------------
// Part B: 2D CUDA Rotation Kernel (Sub-tasks 1a & 1b)
// ----------------------------------------------------------------------------
__global__ void rotate_kernel(const unsigned char* __restrict__ input,
                             unsigned char* __restrict__ output,
                             int width, int height,
                             float cos_a, float sin_a,
                             float center) {
    // 2D CUDA Thread Indexing
    int col = blockIdx.x * blockDim.x + threadIdx.x; // x destination coordinate
    int row = blockIdx.y * blockDim.y + threadIdx.y; // y destination coordinate

    // Boundary check for arbitrary image dimensions
    if (col < width && row < height) {
        float x_shifted = static_cast<float>(col) - center;
        float y_shifted = static_cast<float>(row) - center;

        // Inverse Rotation Coordinate Lookup
        int src_x = static_cast<int>(roundf(x_shifted * cos_a + y_shifted * sin_a + center));
        int src_y = static_cast<int>(roundf(-x_shifted * sin_a + y_shifted * cos_a + center));

        std::size_t dst_offset = (static_cast<std::size_t>(row) * width + col) * 3;

        if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
            std::size_t src_offset = (static_cast<std::size_t>(src_y) * width + src_x) * 3;
            output[dst_offset]     = input[src_offset];     // Red
            output[dst_offset + 1] = input[src_offset + 1]; // Green
            output[dst_offset + 2] = input[src_offset + 2]; // Blue
        } else {
            // Out of bounds: Zero-padding (black background)
            output[dst_offset]     = 0;
            output[dst_offset + 1] = 0;
            output[dst_offset + 2] = 0;
        }
    }
}

// ----------------------------------------------------------------------------
// Part C: Performance Benchmarking & Validation Structure
// ----------------------------------------------------------------------------
struct Result {
    int size;
    double cpu, h2d, kernel, d2h;

    double total_gpu() const {
        return h2d + kernel + d2h;
    }

    double speedup() const {
        return cpu / total_gpu();
    }
};

Result benchmark(int size, float angle_deg, std::mt19937& rng, bool save_images = false) {
    const Image image = make_image(size, rng);
    const std::size_t bytes = image.size() * sizeof(unsigned char);
    const float rad = angle_deg * (3.14159265358979323846f / 180.0f);
    const float cos_a = std::cos(rad);
    const float sin_a = std::sin(rad);
    const float center = (static_cast<float>(size) - 1.0f) / 2.0f;

    Result result{size, 0.0, 0.0, 0.0, 0.0};

    // Measure CPU Reference Execution
    auto start = Clock::now();
    const Image cpu_result = rotate_cpu(image, size, angle_deg);
    result.cpu = elapsed(start);

    // Configure 2D Execution Grid & Block Dimensions
    const dim3 threads_per_block(16, 16);
    const dim3 blocks_per_grid((size + 15) / 16, (size + 15) / 16);

    if (save_images) {
        std::cout << "Image Shape     : (" << size << ", " << size << ", 3 RGB)\n"
                  << "Grid Dimensions : (" << blocks_per_grid.x << ", " << blocks_per_grid.y << ") blocks\n"
                  << "Block Dimensions: (16, 16) threads\n";
    }

    DeviceBuffer d_input, d_output;

    // 1. Host-to-Device Memory Transfer (H2D)
    start = Clock::now();
    d_input.allocate(bytes);
    check(cudaMemcpy(d_input.data, image.data(), bytes, cudaMemcpyHostToDevice), "H2D copy");
    check(cudaDeviceSynchronize(), "Synchronize H2D");
    result.h2d = elapsed(start);

    // Allocate GPU Output Buffer
    d_output.allocate(bytes);
    check(cudaDeviceSynchronize(), "Synchronize before kernel");

    // 2. Parallel CUDA Kernel Execution
    start = Clock::now();
    rotate_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_input.data, d_output.data, size, size, cos_a, sin_a, center);
    check(cudaGetLastError(), "Launch rotate_kernel");
    check(cudaDeviceSynchronize(), "Execute rotate_kernel");
    result.kernel = elapsed(start);

    // 3. Device-to-Host Memory Transfer (D2H)
    check(cudaDeviceSynchronize(), "Synchronize before D2H");
    start = Clock::now();
    Image gpu_result(image.size());
    check(cudaMemcpy(gpu_result.data(), d_output.data, bytes, cudaMemcpyDeviceToHost), "D2H copy");
    check(cudaDeviceSynchronize(), "Synchronize D2H");
    result.d2h = elapsed(start);

    // Validation Check: Verify CPU vs GPU pixel outputs match
    if (cpu_result != gpu_result) {
        throw std::runtime_error("Validation Failed: CPU and GPU rotation outputs differ!");
    }

    if (save_images) {
        save_ppm("original.ppm", image, size);
        save_ppm("rotated_45deg.ppm", gpu_result, size);
        std::cout << "Validation Passed: CPU and GPU outputs match identically.\n"
                  << "Saved PPM files: 'original.ppm' and 'rotated_45deg.ppm'\n";
    }

    return result;
}

// ----------------------------------------------------------------------------
// Main Routine: GPU Query, Warm-up & Benchmark Suite Execution
// ----------------------------------------------------------------------------
int main() {
    try {
        check(cudaSetDevice(0), "Select GPU device 0");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, 0), "Read GPU properties");

        int version = 0;
        check(cudaRuntimeGetVersion(&version), "Read CUDA runtime version");

        std::cout << "====================================================\n"
                  << "GPU Image Rotation Benchmark\n"
                  << "====================================================\n"
                  << "GPU Model            : " << properties.name << "\n"
                  << "Total Global Memory  : " << properties.totalGlobalMem / (1024.0 * 1024.0) << " MiB\n"
                  << "CUDA Runtime Version : " << version / 1000 << '.' << (version % 1000) / 10 << "\n"
                  << "====================================================\n\n";

        // Warm up CUDA context before timing
        check(cudaFree(nullptr), "Initialize CUDA Context");

        std::mt19937 rng(3143); // Fixed seed matching lecture material

        // Detailed single-run benchmark on 1024x1024
        const Result first = benchmark(1024, 45.0f, rng, true);

        std::cout << std::fixed << std::setprecision(6)
                  << "\n===== PERFORMANCE SUMMARY (1024 x 1024) =====\n"
                  << "CPU Reference Time      : " << first.cpu << " s\n"
                  << "Host -> Device Transfer : " << first.h2d << " s\n"
                  << "Kernel Execution Time   : " << first.kernel << " s\n"
                  << "Device -> Host Transfer : " << first.d2h << " s\n"
                  << "Total GPU Execution Time: " << first.total_gpu() << " s\n"
                  << "Overall Speedup         : " << first.speedup() << "x\n\n";

        // Multi-resolution benchmark suite
        std::vector<Result> results;
        for (int size : {512, 1024, 2048, 4096}) {
            std::cout << "Benchmarking Resolution: " << size << " x " << size << " ...\n";
            results.push_back(benchmark(size, 45.0f, rng));
        }

        std::cout << "\n"
                  << std::left << std::setw(10) << "Size"
                  << std::setw(12) << "CPU(ms)"
                  << std::setw(12) << "H2D(ms)"
                  << std::setw(14) << "Kernel(ms)"
                  << std::setw(12) << "D2H(ms)"
                  << std::setw(12) << "GPU(ms)"
                  << std::setw(10) << "Speedup" << "\n"
                  << std::string(82, '-') << "\n"
                  << std::setprecision(3);

        for (const auto& r : results) {
            std::cout << std::setw(10) << r.size
                      << std::setw(12) << r.cpu * 1000
                      << std::setw(12) << r.h2d * 1000
                      << std::setw(14) << r.kernel * 1000
                      << std::setw(12) << r.d2h * 1000
                      << std::setw(12) << r.total_gpu() * 1000
                      << std::setw(10) << r.speedup() << "x\n";
        }

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}

/*
============================================================================
Image Rotation with GPU (CUDA C)
============================================================================
Written in CUDA C, explicit error handling (no try/catch), and POSIX clock timing.

Build with NVIDIA CUDA Toolkit:
  nvcc -O3 image_rotation_c.cu -o image_rotation_c

Run the compiled executable:
  ./image_rotation_c
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#define PI 3.14159265358979323846f

// ----------------------------------------------------------------------------
// Helper: Error Checking Function for CUDA Runtime APIs
// ----------------------------------------------------------------------------
void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA Error during [%s]: %s\n", operation, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

// ----------------------------------------------------------------------------
// Helper: High-Resolution Timer in Seconds (POSIX clock_gettime)
// ----------------------------------------------------------------------------
double get_time_sec(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

// ----------------------------------------------------------------------------
// Helper: Synthetic RGB Image Generation & PPM File Export
// ----------------------------------------------------------------------------
unsigned char* make_image(int size) {
    size_t num_bytes = (size_t)size * size * 3;
    unsigned char* image = (unsigned char*)malloc(num_bytes);
    if (!image) {
        fprintf(stderr, "Allocation failure for host image buffer.\n");
        exit(EXIT_FAILURE);
    }

    // Pseudo-random RGB pixel generation (deterministic seed)
    srand(3143);
    for (size_t i = 0; i < num_bytes; i++) {
        image[i] = (unsigned char)(rand() % 256);
    }
    return image;
}

void save_ppm(const char* filename, const unsigned char* image, int size) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Cannot create PPM file %s\n", filename);
        return;
    }
    fprintf(file, "P6\n%d %d\n255\n", size, size);
    size_t written = fwrite(image, 1, (size_t)size * size * 3, file);
    if (written != (size_t)size * size * 3) {
        fprintf(stderr, "Error: Incomplete write to file %s\n", filename);
    }
    fclose(file);
}

// ----------------------------------------------------------------------------
// Part A: CPU Baseline Reference Rotation Routine
// ----------------------------------------------------------------------------
void rotate_cpu(const unsigned char* input, unsigned char* output, int size, float angle_deg) {
    const float rad = angle_deg * (PI / 180.0f);
    const float cos_a = cosf(rad);
    const float sin_a = sinf(rad);
    const float center = ((float)size - 1.0f) / 2.0f;

    for (int y_dst = 0; y_dst < size; ++y_dst) {
        for (int x_dst = 0; x_dst < size; ++x_dst) {
            float x_shift = (float)x_dst - center;
            float y_shift = (float)y_dst - center;

            // Inverse rotation linear algebra formula
            int src_x = (int)roundf(x_shift * cos_a + y_shift * sin_a + center);
            int src_y = (int)roundf(-x_shift * sin_a + y_shift * cos_a + center);

            size_t dst_offset = ((size_t)y_dst * size + x_dst) * 3;

            if (src_x >= 0 && src_x < size && src_y >= 0 && src_y < size) {
                size_t src_offset = ((size_t)src_y * size + src_x) * 3;
                output[dst_offset]     = input[src_offset];     // Red
                output[dst_offset + 1] = input[src_offset + 1]; // Green
                output[dst_offset + 2] = input[src_offset + 2]; // Blue
            } else {
                output[dst_offset]     = 0;
                output[dst_offset + 1] = 0;
                output[dst_offset + 2] = 0;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Part B: 2D CUDA Rotation Kernel
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
        float x_shifted = (float)col - center;
        float y_shifted = (float)row - center;

        // Inverse Rotation Coordinate Lookup
        int src_x = (int)roundf(x_shifted * cos_a + y_shifted * sin_a + center);
        int src_y = (int)roundf(-x_shifted * sin_a + y_shifted * cos_a + center);

        size_t dst_offset = ((size_t)row * width + col) * 3;

        if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
            size_t src_offset = ((size_t)src_y * width + src_x) * 3;
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
typedef struct {
    int size;
    double cpu;
    double h2d;
    double kernel;
    double d2h;
    double total_gpu;
    double speedup;
} BenchmarkResult;

BenchmarkResult run_benchmark(int size, float angle_deg, int save_images) {
    size_t num_bytes = (size_t)size * size * 3 * sizeof(unsigned char);
    unsigned char* h_input = make_image(size);
    unsigned char* h_cpu_output = (unsigned char*)malloc(num_bytes);
    unsigned char* h_gpu_output = (unsigned char*)malloc(num_bytes);

    if (!h_cpu_output || !h_gpu_output) {
        fprintf(stderr, "Allocation failure for host output buffers.\n");
        exit(EXIT_FAILURE);
    }

    const float rad = angle_deg * (PI / 180.0f);
    const float cos_a = cosf(rad);
    const float sin_a = sinf(rad);
    const float center = ((float)size - 1.0f) / 2.0f;

    BenchmarkResult res;
    res.size = size;

    // 1. Measure CPU Reference Execution
    double start_time = get_time_sec();
    rotate_cpu(h_input, h_cpu_output, size, angle_deg);
    res.cpu = get_time_sec() - start_time;

    // Configure 2D Execution Grid & Block Dimensions
    dim3 threads_per_block;
    threads_per_block.x = 16;
    threads_per_block.y = 16;
    threads_per_block.z = 1;

    dim3 blocks_per_grid;
    blocks_per_grid.x = (size + 15) / 16;
    blocks_per_grid.y = (size + 15) / 16;
    blocks_per_grid.z = 1;

    if (save_images) {
        printf("Image Shape     : (%d, %d, 3 RGB)\n", size, size);
        printf("Grid Dimensions : (%d, %d) blocks\n", blocks_per_grid.x, blocks_per_grid.y);
        printf("Block Dimensions: (16, 16) threads\n");
    }

    unsigned char *d_input = NULL, *d_output = NULL;

    // 2. Allocate Device GPU Memory
    check_cuda(cudaMalloc((void**)&d_input, num_bytes), "cudaMalloc d_input");
    check_cuda(cudaMalloc((void**)&d_output, num_bytes), "cudaMalloc d_output");

    // 3. Host-to-Device Memory Transfer (H2D)
    start_time = get_time_sec();
    check_cuda(cudaMemcpy(d_input, h_input, num_bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    check_cuda(cudaDeviceSynchronize(), "Synchronize H2D");
    res.h2d = get_time_sec() - start_time;

    // 4. Parallel CUDA Kernel Execution
    start_time = get_time_sec();
    rotate_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_input, d_output, size, size, cos_a, sin_a, center);
    check_cuda(cudaGetLastError(), "Launch rotate_kernel");
    check_cuda(cudaDeviceSynchronize(), "Execute rotate_kernel");
    res.kernel = get_time_sec() - start_time;

    // 5. Device-to-Host Memory Transfer (D2H)
    start_time = get_time_sec();
    check_cuda(cudaMemcpy(h_gpu_output, d_output, num_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    check_cuda(cudaDeviceSynchronize(), "Synchronize D2H");
    res.d2h = get_time_sec() - start_time;

    res.total_gpu = res.h2d + res.kernel + res.d2h;
    res.speedup = res.cpu / res.total_gpu;

    // 6. Validation Check: Verify CPU vs GPU outputs match
    if (memcmp(h_cpu_output, h_gpu_output, num_bytes) != 0) {
        fprintf(stderr, "Validation Failed: CPU and GPU rotation outputs differ!\n");
        exit(EXIT_FAILURE);
    }

    if (save_images) {
        save_ppm("original_c.ppm", h_input, size);
        save_ppm("rotated_45deg_c.ppm", h_gpu_output, size);
        printf("Validation Passed: CPU and GPU outputs match identically.\n");
        printf("Saved PPM files: 'original_c.ppm' and 'rotated_45deg_c.ppm'\n");
    }

    // Clean up Host & Device memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_cpu_output);
    free(h_gpu_output);

    return res;
}

// ----------------------------------------------------------------------------
// Main Routine: GPU Query, Warm-up & Benchmark Suite Execution
// ----------------------------------------------------------------------------
int main(void) {
    check_cuda(cudaSetDevice(0), "Select GPU device 0");
    struct cudaDeviceProp properties;
    check_cuda(cudaGetDeviceProperties(&properties, 0), "Read GPU properties");

    int version = 0;
    check_cuda(cudaRuntimeGetVersion(&version), "Read CUDA runtime version");

    printf("====================================================\n");
    printf("GPU Image Rotation Benchmark\n");
    printf("====================================================\n");
    printf("GPU Model            : %s\n", properties.name);
    printf("Total Global Memory  : %.2f MiB\n", properties.totalGlobalMem / (1024.0 * 1024.0));
    printf("CUDA Runtime Version : %d.%d\n", version / 1000, (version % 1000) / 10);
    printf("====================================================\n\n");

    // Warm up CUDA context before timing
    check_cuda(cudaFree(0), "Initialize CUDA Context");

    // Detailed single-run benchmark on 1024x1024
    BenchmarkResult first = run_benchmark(1024, 45.0f, 1);

    printf("\n===== PERFORMANCE SUMMARY (1024 x 1024) =====\n");
    printf("CPU Reference Time      : %.6f s\n", first.cpu);
    printf("Host -> Device Transfer : %.6f s\n", first.h2d);
    printf("Kernel Execution Time   : %.6f s\n", first.kernel);
    printf("Device -> Host Transfer : %.6f s\n", first.d2h);
    printf("Total GPU Execution Time: %.6f s\n", first.total_gpu);
    printf("Overall Speedup         : %.2fx\n\n", first.speedup);

    // Multi-resolution benchmark suite
    int sizes[] = {512, 1024, 2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("%-10s %-12s %-12s %-14s %-12s %-12s %-10s\n",
           "Size", "CPU(ms)", "H2D(ms)", "Kernel(ms)", "D2H(ms)", "GPU(ms)", "Speedup");
    printf("----------------------------------------------------------------------------------\n");

    for (int i = 0; i < num_sizes; i++) {
        printf("Benchmarking Resolution: %d x %d ...\n", sizes[i], sizes[i]);
        BenchmarkResult r = run_benchmark(sizes[i], 45.0f, 0);
        printf("%-10d %-12.3f %-12.3f %-14.3f %-12.3f %-12.3f %-10.2fx\n",
               r.size, r.cpu * 1000.0, r.h2d * 1000.0, r.kernel * 1000.0,
               r.d2h * 1000.0, r.total_gpu * 1000.0, r.speedup);
    }

    return 0;
}

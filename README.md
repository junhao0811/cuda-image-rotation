# 🖼️ GPU Image Rotation in CUDA

Rotates RGB images on the GPU with one CUDA thread per output pixel, then benchmarks it against a CPU version from 512×512 up to 4096×4096. The headline result is a classic GPU lesson: **the kernel is up to 98× faster, but moving the data over PCIe eats most of the gain.**

---

## ⚙️ How it works

- **One thread per pixel.** A 2D grid of thread blocks covers the output image (`blockIdx`, `threadIdx` → *(x, y)*).
- **Inverse mapping.** Each thread rotates its destination coordinate *backwards* around the image centre to find the source pixel, so every output pixel is written exactly once, with no gaps or collisions.
- **Nearest-neighbour sampling** with zero padding (black) for pixels that map outside the original image.
- **Full pipeline timing.** Host→device copy, kernel and device→host copy are each timed separately with high-resolution host timers and `cudaDeviceSynchronize`, after a warm-up run so driver start-up doesn't skew results.
- Saves the original and rotated images as `.ppm` files.

Two equivalent implementations are included: `image_rotation.cu` (CUDA C++) and `image_rotation_c.cu` (CUDA C, explicit error handling).

---

## 📊 Benchmark results

| Resolution | CPU (ms) | GPU kernel (ms) | PCIe transfers (ms) | Kernel speed-up | Overall speed-up | Time spent on PCIe |
|---|---|---|---|---|---|---|
| 512 × 512 | 1.73 | 0.35 | 0.90 | 5.0× | 1.4× | 72% |
| 1024 × 1024 | 7.63 | 0.37 | 2.39 | 20.7× | 2.8× | 87% |
| 2048 × 2048 | 29.7 | 0.50 | 8.23 | 59.6× | 3.4× | 94% |
| 4096 × 4096 | 106.4 | 1.08 | 27.0 | **98.3×** | **3.8×** | **96%** |

**Takeaways**
- Kernel speed-up grows with image size, since bigger images keep more GPU cores busy.
- But the **host↔device transfer dominates**, at up to 96% of total GPU time, so end-to-end speed-up stays under 4×. Batching transfers or overlapping them with compute (CUDA streams) is the way forward.
- The **first** run is about 44× slower than the rest because of CUDA driver start-up, which is why the benchmark warms up first.

Raw data: [`docs/benchmark_results.xlsx`](docs/benchmark_results.xlsx). Full write-up: [`docs/report.pdf`](docs/report.pdf).

---

## 🚀 Build and run

Requires an NVIDIA GPU and the CUDA Toolkit (works on Google Colab with a GPU runtime).

```bash
nvcc -O3 -std=c++14 image_rotation.cu -o image_rotation && ./image_rotation
nvcc -O3 image_rotation_c.cu -o image_rotation_c     && ./image_rotation_c
```

## 👥 Team
Built in a team of three.

## 🛠️ Tech
CUDA C/C++ · GPU memory management · performance analysis

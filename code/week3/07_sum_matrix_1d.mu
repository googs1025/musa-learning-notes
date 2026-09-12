// 一维 grid + 一维 block：矩阵按行求和
//
// 这份代码和 06_sum_matrix_2d.mu 计算同一个问题，但线程只有一个线性编号：
//
//   idx = blockIdx.x * blockDim.x + threadIdx.x
//   x = idx % width
//   y = idx / width
//   rows[y] += m[idx]
//
// 也就是说，二维坐标 (x,y) 只是被“压平”成 idx 后再恢复出来。

#include "musa_common.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

// 输入矩阵是 row-major：第 y 行第 x 列位于 m[y * width + x]。
// 一维 kernel 直接使用 idx 访问 m[idx]，再通过除法和取模恢复行列坐标。
//
// 以 width=8、blockDim.x=4 为例：
//
//   idx:  0  1  2  3 | 4  5  6  7 | 8  9 10 11
//   x:    0  1  2  3 | 4  5  6  7 | 0  1  2  3
//   y:    0  0  0  0 | 0  0  0  0 | 1  1  1  1
//
// 实际计算中：x = idx % width，y = idx / width；rows[y] 是第 y 行的和。
__global__ void matrix_to_row_sums_1d(const float* m, float* rows,
                                       int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    int x = idx % width;
    int y = idx / width;
    // x 主要用于展示二维坐标恢复；m[idx] 与 m[y * width + x] 等价。
    (void)x;

    // 同一行的多个线程会同时更新 rows[y]，因此需要 atomicAdd。
    atomicAdd(&rows[y], m[idx]);
}

struct ErrorStats {
    double max_abs;
    double max_rel;
};

static ErrorStats compare_rows(const double* reference, const float* actual,
                               int count) {
    ErrorStats stats{0.0, 0.0};
    for (int i = 0; i < count; ++i) {
        double abs_error = std::fabs(reference[i] - static_cast<double>(actual[i]));
        double scale = std::fmax(std::fabs(reference[i]), 1e-12);
        stats.max_abs = std::fmax(stats.max_abs, abs_error);
        stats.max_rel = std::fmax(stats.max_rel, abs_error / scale);
    }
    return stats;
}

int main() {
    const int W = 1024;
    const int H = 1024;
    const int total = W * H;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    const size_t matrix_bytes = static_cast<size_t>(total) * sizeof(float);
    const size_t rows_bytes = static_cast<size_t>(H) * sizeof(float);

    float* h = static_cast<float*>(std::malloc(matrix_bytes));
    double* rows_cpu = static_cast<double*>(std::calloc(H, sizeof(double)));
    float* rows_gpu = static_cast<float*>(std::calloc(H, sizeof(float)));
    float* d = nullptr;
    float* d_rows = nullptr;
    MUSA_CHECK(musaMalloc(&d, matrix_bytes));
    MUSA_CHECK(musaMalloc(&d_rows, rows_bytes));

    // 使用固定的非全 1 输入，让 CPU double 与 GPU float 的累加差异可观察。
    for (int i = 0; i < total; ++i) {
        h[i] = 1.0f + static_cast<float>(i % 17) * 0.03125f;
    }

    CpuTimer cpu_timer;
    cpu_timer.start();
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            rows_cpu[y] += static_cast<double>(h[y * W + x]);
        }
    }
    double cpu_ms = cpu_timer.elapsed_ms();

    // kernel-only：只测 GPU kernel，不包含 H2D/D2H。
    MUSA_CHECK(musaMemcpy(d, h, matrix_bytes, musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemset(d_rows, 0, rows_bytes));
    GpuTimer kernel_timer;
    kernel_timer.start();
    matrix_to_row_sums_1d<<<blocks, threads>>>(d, d_rows, W, H);
    kernel_timer.stop();
    float gpu_kernel_ms = kernel_timer.elapsed_ms();
    MUSA_CHECK_KERNEL();
    MUSA_CHECK(musaMemcpy(rows_gpu, d_rows, rows_bytes, musaMemcpyDeviceToHost));

    // end-to-end：重新执行一次，测 H2D + kernel + 同步 + D2H。
    MUSA_CHECK(musaMemset(d_rows, 0, rows_bytes));
    CpuTimer gpu_wall_timer;
    gpu_wall_timer.start();
    MUSA_CHECK(musaMemcpy(d, h, matrix_bytes, musaMemcpyHostToDevice));
    matrix_to_row_sums_1d<<<blocks, threads>>>(d, d_rows, W, H);
    MUSA_CHECK(musaDeviceSynchronize());
    MUSA_CHECK(musaMemcpy(rows_gpu, d_rows, rows_bytes, musaMemcpyDeviceToHost));
    double gpu_end_to_end_ms = gpu_wall_timer.elapsed_ms();

    double total_cpu = 0.0;
    double total_gpu = 0.0;
    for (int y = 0; y < H; ++y) {
        total_cpu += rows_cpu[y];
        total_gpu += static_cast<double>(rows_gpu[y]);
    }
    ErrorStats row_error = compare_rows(rows_cpu, rows_gpu, H);
    double total_abs_error = std::fabs(total_cpu - total_gpu);
    double total_rel_error = total_abs_error / std::fmax(std::fabs(total_cpu), 1e-12);
    // float atomicAdd 的累加顺序不确定，使用适当的绝对/相对误差阈值。
    constexpr double abs_tol = 1e-2;
    constexpr double rel_tol = 1e-5;
    bool pass = row_error.max_abs <= abs_tol && row_error.max_rel <= rel_tol
             && total_abs_error <= abs_tol && total_rel_error <= rel_tol;

    std::printf("[config] matrix=%d x %d, 1D block=%d, 1D grid=%d\n",
                W, H, threads, blocks);
    std::printf("[time] CPU=%.4f ms, GPU kernel-only=%.4f ms, GPU end-to-end=%.4f ms\n",
                cpu_ms, gpu_kernel_ms, gpu_end_to_end_ms);
    for (int y = 0; y < H; ++y) {
        double abs_error = std::fabs(rows_cpu[y] - static_cast<double>(rows_gpu[y]));
        std::printf("row[%d] CPU=%.9f GPU=%.9f abs_error=%.12e\n",
                    y, rows_cpu[y], rows_gpu[y], abs_error);
    }
    std::printf("[rows] CPU row0=%.9f row_last=%.9f\n",
                rows_cpu[0], rows_cpu[H - 1]);
    std::printf("[rows] GPU row0=%.9f row_last=%.9f\n",
                rows_gpu[0], rows_gpu[H - 1]);
    std::printf("[total] CPU=%.12f GPU=%.12f abs_error=%.12e rel_error=%.12e\n",
                total_cpu, total_gpu, total_abs_error, total_rel_error);
    std::printf("[error] max_row_abs=%.12e max_row_rel=%.12e tolerance=(%.1e, %.1e) %s\n",
                row_error.max_abs, row_error.max_rel, abs_tol, rel_tol,
                pass ? "PASS" : "FAIL");

    MUSA_CHECK(musaFree(d));
    MUSA_CHECK(musaFree(d_rows));
    std::free(h);
    std::free(rows_cpu);
    std::free(rows_gpu);
    return pass ? 0 : 1;
}

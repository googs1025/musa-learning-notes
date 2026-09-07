// ============================================================================
//  示例: device 侧 pow 类数学函数
//
//  学习目标:
//    1. 通过 USE_FP32 切换 float/double 版本的幂函数。
//    2. 理解每个线程对一个元素执行 a[idx]^2 + b[idx]。
//    3. 观察数学函数的类型选择会影响精度和开销。
//
//  注意:
//    如果指数固定为 2, 真实性能敏感代码里通常优先写 a[idx] * a[idx]。这里使用
//    powi/powif 是为了演示 MUSA 数学函数调用。
//
//  阅读顺序:
//    先看 USE_FP32 对 powt/MUSA_POW 的选择, 再看线程索引、边界条件和 host/device 数据流。
// ============================================================================
#ifndef USE_FP32
#define MUSA_POW(x, y) powi(x, y)
typedef double powt;
#else
#define MUSA_POW(x, y) powif(x, y)
typedef float powt;
#endif

__global__ void VectorPow(powt *a, uint32_t *b) {
    // 一个线程负责一个数组元素; 当前示例的 launch 参数正好覆盖数据, 通用写法仍应增加 idx < n 边界保护。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] = MUSA_POW(a[idx], 2) + b[idx];
}

const size_t numElements = 1048576;
const size_t sizeBytesA = numElements * sizeof(powt);
const size_t sizeBytesB = numElements * sizeof(uint32_t);
const int threadsPerBlock = 1024;
const int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

int main() {
    powt *hA = nullptr, *dA = nullptr;
    uint32_t *hB = nullptr, *dB = nullptr;

    hA = reinterpret_cast<powt *>(malloc(sizeBytesA));
    hB = reinterpret_cast<uint32_t *>(malloc(sizeBytesB));

    assert(hA != nullptr && hB != nullptr && "host malloc failed!");

    for (uint32_t i = 0; i < numElements; ++i) {
        hA[i] = (powt)i;
        hB[i] = 2 * i;
    }

    checkMusaErrors(musaMalloc(&dA, sizeBytesA));
    checkMusaErrors(musaMalloc(&dB, sizeBytesB));

    checkMusaErrors(musaMemcpy(dA, hA, sizeBytesA, musaMemcpyHostToDevice));
    checkMusaErrors(musaMemcpy(dB, hB, sizeBytesB, musaMemcpyHostToDevice));

    // kernel launch 本身通常是异步入队, 后面的 synchronize 才是观察结果前的等待点。
    VectorPow<<<blocksPerGrid, threadsPerBlock>>>(dA, dB);

    checkMusaErrors(musaDeviceSynchronize());
    checkMusaErrors(musaMemcpy(hA, dA, sizeBytesA, musaMemcpyDeviceToHost));

    free(hA);
    free(hB);
    checkMusaErrors(musaFree(dA));
    checkMusaErrors(musaFree(dB));
    return 0;
}

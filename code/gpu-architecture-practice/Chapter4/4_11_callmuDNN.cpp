// ============================================================================
//  示例: muDNN API 调用骨架
//
//  学习目标:
//    1. 理解深度学习库的 descriptor 模型: tensor/filter/conv 等元数据先声明。
//    2. 区分输入输出数据、权重、workspace 和算法选择。
//    3. 用统一 CHECK 宏尽早暴露 runtime 与 muDNN 错误。
//
//  阅读顺序:
//    先看 descriptor 的创建和 shape 配置, 再看 workspace、算法选择和执行调用。
//
//  注意:
//    descriptor 的 shape、layout、dtype 必须和真实内存一致, 否则问题通常不会在
//    编译期暴露, 只能在运行期或结果校验时发现。
// ============================================================================
#include <random>
#include "mudnn.h"
#include "musa_fp16.h"
#include "musa_runtime.h"
// 检查MUSA运行时状态码
#define MUSA_CHECK(expr)                                                                           \
    do {                                                                                           \
        musaError_t _err = (expr);                                                                 \
        if (_err != musaSuccess) {                                                                 \
            fprintf(stderr, "musa runtime failed " #expr ": err=%d(%s) line=%s:%d\n", (int)_err,   \
                    musaGetErrorString(_err), __FILE__, __LINE__);                                 \
            abort();                                                                               \
        }                                                                                          \
    } while (0)
// 检查muDNN状态码
#define DNN_CHECK(expr)                                                                            \
    do {                                                                                           \
        if (expr != ::musa::dnn::Status::SUCCESS) {                                                \
            fprintf(stderr, "muDNN failed " #expr ": line=%s:%d\n", __FILE__, __LINE__);           \
            abort();                                                                               \
        }                                                                                          \
    } while (0)
using namespace ::musa::dnn;
using float16_t = __half;
// 生成随机数函数
void GenerateRandom(float16_t *data, int64_t size, uint seed = 2333) {
    std::default_random_engine engine(seed);
    std::uniform_real_distribution<float> dist(-1, 1);
    for (auto i = 0; i < size; i++) {
        data[i] = (float16_t)(dist(engine));
    }
}
// 创建自定义设备端内存释放函数
void MemFree(void *ptr) {
    if (ptr != nullptr) {
        MUSA_CHECK(musaFree(ptr));
    }
}
// 自定义设备端内存分配方式
MemoryHandler MemoryFunc(size_t size) {
    void *data = nullptr;
    MUSA_CHECK(musaMalloc(&data, size));
    return MemoryHandler(data, MemFree);
}
int main(int argc, char *argv[]) {
    // 创建并初始化muDNN Handle，并指定计算流
    musaStream_t stream;
    MUSA_CHECK(musaStreamCreate(&stream));
    Handle handle(0);
    handle.SetStream(stream);
    // 指定批量矩阵乘法问题规模和转置
    int batch_size = 4;
    int m = 1024;
    int n = 2048;
    int k = 4096;
    bool trans_a = false; // A不转置
    bool trans_b = true;  // B转置
    size_t nr_elem_a = (size_t)batch_size * m * k;
    size_t nr_elem_b = (size_t)batch_size * k * n;
    size_t nr_elem_c = (size_t)batch_size * m * n;
    size_t nr_elem_d = nr_elem_c;
    size_t nr_elem_bias = (size_t)n;
    size_t size_a = nr_elem_a * sizeof(float16_t);
    size_t size_b = nr_elem_b * sizeof(float16_t);
    size_t size_c = nr_elem_c * sizeof(float16_t);
    size_t size_d = nr_elem_d * sizeof(float16_t);
    size_t size_bias = nr_elem_bias * sizeof(float16_t);
    // 分配主机端内存
    void *h_buff_a = (void *)malloc(size_a);
    void *h_buff_b = (void *)malloc(size_b);
    void *h_buff_c = (void *)malloc(size_c);
    void *h_buff_d = (void *)malloc(size_d);
    void *h_buff_bias = (void *)malloc(size_bias);
    if (!h_buff_a || !h_buff_b || !h_buff_c || !h_buff_d || !h_buff_bias) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }
    // 生成主机端随机数据
    GenerateRandom((float16_t *)h_buff_a, nr_elem_a);
    GenerateRandom((float16_t *)h_buff_b, nr_elem_b);
    GenerateRandom((float16_t *)h_buff_c, nr_elem_c);
    GenerateRandom((float16_t *)h_buff_d, nr_elem_d);
    GenerateRandom((float16_t *)h_buff_bias, nr_elem_bias);
    // 分配设备端内存
    void *d_buff_a = nullptr;
    void *d_buff_b = nullptr;
    void *d_buff_c = nullptr;
    void *d_buff_d = nullptr;
    void *d_buff_bias = nullptr;
    MUSA_CHECK(musaMalloc(&d_buff_a, size_a));
    MUSA_CHECK(musaMalloc(&d_buff_b, size_b));
    MUSA_CHECK(musaMalloc(&d_buff_c, size_c));
    MUSA_CHECK(musaMalloc(&d_buff_d, size_d));
    MUSA_CHECK(musaMalloc(&d_buff_bias, size_bias));
    // 将主机端数据拷贝到设备端
    MUSA_CHECK(musaMemcpy(d_buff_a, h_buff_a, size_a, musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_buff_b, h_buff_b, size_b, musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_buff_c, h_buff_c, size_c, musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_buff_d, h_buff_d, size_d, musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_buff_bias, h_buff_bias, size_bias, musaMemcpyHostToDevice));
    // 创建muDNN Tensor并设置其属性
    Tensor tensor_a, tensor_b, tensor_c, tensor_d, tensor_bias;
    Tensor::Type ttype = Tensor::Type::HALF;
    tensor_d.SetAddr(d_buff_d);
    tensor_d.SetType(ttype);
    tensor_d.SetNdInfo({batch_size, m, n});
    tensor_a.SetAddr(d_buff_a);
    tensor_a.SetType(ttype);
    if (trans_a) {
        tensor_a.SetNdInfo({batch_size, k, m});
    } else {
        tensor_a.SetNdInfo({batch_size, m, k});
    }
    tensor_b.SetAddr(d_buff_b);
    tensor_b.SetType(ttype);
    if (trans_b) {
        tensor_b.SetNdInfo({batch_size, n, k});
    } else {
        tensor_b.SetNdInfo({batch_size, k, n});
    }
    tensor_c.SetAddr(d_buff_c);
    tensor_c.SetType(ttype);
    tensor_c.SetNdInfo({batch_size, m, n});
    // 创建偏置Tensor
    tensor_bias.SetAddr(d_buff_bias);
    tensor_bias.SetType(ttype);
    tensor_bias.SetNdInfo({n});
    MUSA_CHECK(musaStreamSynchronize(stream)); // 确保流同步
    MUSA_CHECK(musaDeviceSynchronize());       // 确保设备同步
    // 创建批量矩阵乘法算子
    BatchMatMul bmm_op;
    MatMulLtParam lt_param;
    DNN_CHECK(bmm_op.SetTranspose(trans_a, trans_b));              // 设置转置
    DNN_CHECK(bmm_op.SetComputeMode(MatMul::ComputeMode::TENSOR)); // 设置计算模式为TENSOR
    DNN_CHECK(bmm_op.SetSplitK(false));                            // 不使用SplitK
    DNN_CHECK(bmm_op.SetAlpha(1.0));                               // 设置alpha为1
    DNN_CHECK(bmm_op.SetBeta(0.0));                                // 设置beta为0
    DNN_CHECK(bmm_op.SetGamma(1.0));                               // 设置gamma为1
    // 执行批量矩阵乘法
    DNN_CHECK(bmm_op.RunLt(handle, tensor_d, tensor_a, tensor_b, tensor_c, tensor_bias, lt_param,
                           MemoryFunc));
    MUSA_CHECK(musaGetLastError()); // 检查是否有错误发生
    // 创建一元操作Unary算子
    Unary unary_op;
    DNN_CHECK(unary_op.SetMode(Unary::Mode::LEAKY_RELU)); // 设置一元操作为Leaky ReLU
    DNN_CHECK(unary_op.SetAlpha(0.01));                   // 设置Leaky ReLU的alpha为0.01
    DNN_CHECK(unary_op.SetBeta(0.0));                     // 设置beta为0
    // 执行一元操作
    DNN_CHECK(unary_op.Run(handle, tensor_d, tensor_d));
    MUSA_CHECK(musaGetLastError()); // 检查是否有错误发生
    // 将结果从设备端拷贝回主机端
    MUSA_CHECK(musaMemcpyAsync(h_buff_d, d_buff_d, size_d, musaMemcpyDeviceToHost, stream));
    MUSA_CHECK(musaStreamSynchronize(stream)); // 等待流同步
    // 打印前10个结果
    printf("Result (first 10 elements):\n");
    for (int i = 0; i < 10 && i < nr_elem_d; i++) {
        printf("%f ", (float)(*((float16_t *)h_buff_d + i * sizeof(float16_t))));
    }
    printf("\n");
    // 释放设备端内存
    MUSA_CHECK(musaFree(d_buff_a));
    MUSA_CHECK(musaFree(d_buff_b));
    MUSA_CHECK(musaFree(d_buff_c));
    MUSA_CHECK(musaFree(d_buff_d));
    MUSA_CHECK(musaFree(d_buff_bias));
    // 释放主机端内存
    free(h_buff_a);
    free(h_buff_b);
    free(h_buff_c);
    free(h_buff_d);
    free(h_buff_bias);
    MUSA_CHECK(musaStreamDestroy(stream)); // 销毁流
    return 0;
}

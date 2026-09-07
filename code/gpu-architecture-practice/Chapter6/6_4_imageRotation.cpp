// ============================================================================
//  示例片段: 使用 texture object 做图像旋转采样
//
//  学习目标:
//    1. 用二维线程网格把每个输出像素映射到一个 GPU 线程。
//    2. 以图像中心为原点做反向坐标变换, 从输入 texture 采样。
//    3. 理解 texture object 适合带空间局部性的只读图像访问。
//
//  阅读顺序:
//    先看 x/y 到输出像素的映射, 再看中心坐标变换、边界判断和 texture 采样。
//
//  注意:
//    这个文件展示 kernel 核心逻辑, texture object 的创建、输入图像绑定和输出
//    buffer 分配需要在完整 host 代码中完成。
// ============================================================================
// 图像旋转核函数
__global__ void rotateImageKernel(musaTextureObject_t texObj, uchar4 *output, int width, int height,
                                  float cosTheta, float sinTheta) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // 计算中心坐标: 后续把输出像素转到以中心为原点的坐标系。
    float cx = width * 0.5f;
    float cy = height * 0.5f;

    // 计算相对中心的偏移
    float dx = (x - cx) * cosTheta - (y - cy) * sinTheta;
    float dy = (x - cx) * sinTheta + (y - cy) * cosTheta;

    // 转换为原始图像坐标
    float srcX = cx + dx;
    float srcY = cy + dy;

    // 归一化纹理坐标
    float u = srcX / width;
    float v = srcY / height;

    // 边界检查
    if (u >= 0.0f && u < 1.0f && v >= 0.0f && v < 1.0f) {
        // 带插值的纹理获取
        output[y * width + x] = tex2D<uchar4>(texObj, u, v);
    } else {
        // 边界外填充黑色
        output[y * width + x] = make_uchar4(0, 0, 0, 255);
    }
}
// 主机端调用代码示例
void rotateImage(uchar4 *d_input, uchar4 *d_output, int width, int height, float angle) {
    // 创建MUSA数组
    musaArray *cuArray;
    musaChannelFormatDesc channelDesc = musaCreateChannelDesc<uchar4>();
    musaMallocArray(&cuArray, &channelDesc, width, height);

    // 复制数据到MUSA数组
    musaMemcpy2DToArray(cuArray, 0, 0, d_input, width * sizeof(uchar4), width * sizeof(uchar4),
                        height, musaMemcpyDeviceToDevice);

    // 配置纹理
    musaResourceDesc resDesc = {};
    resDesc.resType = musaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    musaTextureDesc texDesc = {};
    texDesc.addressMode[0] = musaAddressModeClamp;
    texDesc.addressMode[1] = musaAddressModeClamp;
    texDesc.filterMode = musaFilterModeLinear;
    texDesc.readMode = musaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // 创建纹理对象
    musaTextureObject_t texObj;
    musaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    // 计算旋转参数
    float rad = angle * (M_PI / 180.0f);
    float cosTheta = cosf(rad);
    float sinTheta = sinf(rad);

    // 设置线程块和网格
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // 启动核函数
    rotateImageKernel<<<grid, block>>>(texObj, d_output, width, height, cosTheta, sinTheta);

    // 清理资源
    musaDestroyTextureObject(texObj);
    musaFreeArray(cuArray);
}

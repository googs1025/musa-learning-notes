// ============================================================================
//  示例: MUSA Driver API 虚拟内存管理
//
//  学习目标:
//    1. 区分 runtime API 的 musaMalloc 和 driver API 的地址空间/物理内存管理。
//    2. 通过 muMemAddressReserve 预留虚拟地址, 再用 muMemCreate 创建物理内存。
//    3. 用 muMemMap / muMemSetAccess 建立映射和访问权限。
//
//  阅读顺序:
//    按 reserve -> create -> map -> set access -> 使用 -> unmap/release 的生命周期阅读。
//
//  注意:
//    虚拟内存 API 更底层, 适合大内存池、稀疏映射或自定义 allocator。普通示例
//    优先使用 musaMalloc, 更容易写对。
// ============================================================================
#include <musa.h>
#include <stdio.h>
#include <vector>
#define MUSA_CHECK(call)                                                                           \
    do {                                                                                           \
        MUresult err = (call);                                                                     \
        if (err != MUSA_SUCCESS) {                                                                 \
            const char *errStr;                                                                    \
            muGetErrorString(err, &errStr);                                                        \
            fprintf(stderr, "MUSA error at %s:%d - %s\n", __FILE__, __LINE__, errStr);             \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)
int main() {
    // 1. 初始化MUSA
    int value = 32;
    int ret = 1;
    MUdevice device;
    MUcontext context;
    MUSA_CHECK(muInit(0));
    MUSA_CHECK(muDeviceGet(&device, 0));
    MUSA_CHECK(muCtxCreate(&context, 0, device));
    // 2. 创建物理内存分配。
    // granularity 是驱动要求的最小对齐粒度, 分配大小必须向上取整到该粒度。
    size_t alloc_size = 4 * 1024 * 1024; // 4MB
    size_t granularity = 0;
    MUmemAllocationProp prop = {};
    prop.type = MU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = MU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    MUSA_CHECK(
        muMemGetAllocationGranularity(&granularity, &prop, MU_MEM_ALLOC_GRANULARITY_MINIMUM));
    alloc_size = ((alloc_size + granularity - 1) / granularity) * granularity;
    MUmemGenericAllocationHandle handle;
    MUSA_CHECK(muMemCreate(&handle, alloc_size, &prop, 0));
    // 3. 保留虚拟地址空间。
    // 这一步只拿到一段地址范围, 还没有把真实物理显存挂上去。
    MUdeviceptr va_ptr;
    MUSA_CHECK(muMemAddressReserve(&va_ptr, alloc_size, 0, 0, 0));
    // 4. 映射到虚拟地址。
    // handle 代表物理内存, va_ptr 代表虚拟地址; muMemMap 建立二者关系。
    MUSA_CHECK(muMemMap(va_ptr, alloc_size, 0, handle, 0));
    // 5. 设置访问属性。
    // 没有访问权限时, 地址已经映射也不代表 kernel/拷贝可以读写。
    MUmemAccessDesc accessDesc = {};
    accessDesc.location.type = MU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = MU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    MUSA_CHECK(muMemSetAccess(va_ptr, alloc_size, &accessDesc, 1));
    // 6. 使用内存并校验。
    // Driver API 使用 muMemcpyHtoD/muMemcpyDtoH, 指针类型是 MUdeviceptr。
    std::vector<int> host_data(alloc_size / sizeof(int), value);
    MUSA_CHECK(muMemcpyHtoD(va_ptr, host_data.data(), alloc_size));
    std::vector<int> host_verify(alloc_size / sizeof(int));
    MUSA_CHECK(muMemcpyDtoH(host_verify.data(), va_ptr, alloc_size));
    for (auto data : host_verify) {
        if (data != value) {
            ret = 0;
            printf("Result Check Fail\n");
            break;
        }
    }
    // 7. 清理资源。顺序通常是先 unmap, 再 release 物理内存 handle,
    // 最后释放虚拟地址范围和上下文。
    MUSA_CHECK(muMemUnmap(va_ptr, alloc_size));
    MUSA_CHECK(muMemRelease(handle));
    MUSA_CHECK(muMemAddressFree(va_ptr, alloc_size));
    MUSA_CHECK(muCtxDestroy(context));

    if (ret) {
        printf("PASS!\n");
    }
}

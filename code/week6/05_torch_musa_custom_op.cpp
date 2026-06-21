// torch_musa custom op skeleton. Build against the torch_musa extension helpers for the local SDK.
#include <torch/extension.h>

// 这是最小 C++ 绑定骨架：先验证 PyTorch extension 编译链路，再替换成真正的 MUSA kernel。
torch::Tensor identity_musa(torch::Tensor x) {
    TORCH_CHECK(x.is_contiguous(), "expected contiguous tensor");

    // clone() 保证返回新 Tensor，便于 Python 侧测试是否真的调用了 custom op。
    return x.clone();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("identity_musa", &identity_musa, "identity custom op skeleton");
}

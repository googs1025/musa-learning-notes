// torch_musa custom op skeleton. Build against the torch_musa extension helpers for the local SDK.
#include <torch/extension.h>

torch::Tensor identity_musa(torch::Tensor x) {
    TORCH_CHECK(x.is_contiguous(), "expected contiguous tensor");
    return x.clone();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("identity_musa", &identity_musa, "identity custom op skeleton");
}

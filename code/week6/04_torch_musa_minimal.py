import torch
import torch_musa

# 必须先 import torch_musa，它会把 `musa` 后端注册进 PyTorch。
print("torch", torch.__version__)
print("musa available", torch.musa.is_available())

if torch.musa.is_available():
    # 最小 smoke test：创建 MUSA tensor，跑一个 elementwise op，再同步。
    x = torch.randn(1024, device="musa")
    y = x.relu()
    torch.musa.synchronize()

    # 拷回一个标量用于确认结果链路可用；不要用这个做性能数据。
    print("sample", float(y[0].cpu()))

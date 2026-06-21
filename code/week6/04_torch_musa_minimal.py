import torch
import torch_musa

print("torch", torch.__version__)
print("musa available", torch.musa.is_available())
if torch.musa.is_available():
    x = torch.randn(1024, device="musa")
    y = x.relu()
    torch.musa.synchronize()
    print("sample", float(y[0].cpu()))

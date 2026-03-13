import torch

if torch.cuda.is_available():
    print("GPU")
else:
    print("CPU")

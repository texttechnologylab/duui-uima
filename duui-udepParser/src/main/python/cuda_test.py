import torch

print('CUDA' if torch.cuda.is_available() else 'CPU')
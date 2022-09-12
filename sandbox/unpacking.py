import torch
import numpy as np


class RawToTensorFloat32(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, sample: bytes):
        if isinstance(sample, torch.Tensor):
            return sample
        return torch.from_numpy(np.frombuffer(sample, dtype=np.int16).astype(np.float32) / 2**15)

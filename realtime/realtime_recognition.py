import sounddevice
from audiochains.streams import InputStream, StreamFromFile
from audiochains.block_methods import UnpackRawInInt16, UnpackRawInFloat32

import torch

if __name__ == "__main__":
    with StreamFromFile(
            filename='../data/SA1.WAV.wav',
            blocksize=1024,
    ) as stream:
        stream.set_methods(
            UnpackRawInFloat32()
        )
        for _ in range(stream.get_iterations()):
            chunk = stream.apply()
            chunk = torch.from_numpy(chunk.copy())
            print(chunk)
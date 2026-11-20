import torch
import numpy as np


data = [[1,2], [3,4]]
tensor1 = torch.tensor(data)

data2 = np.array([[5,6], [7,8], [9, 10]])
tensor2 = torch.from_numpy(data2)

data_ones = torch.ones_like(tensor1)

print(data2[1][:])
print(data2.ndim)
print(data2.dtype)
print(data2.shape)

if torch.accelerator.is_available():
    print("We have an accelerator!")
else:
    print("Sorry, CPU only.")
    
print("CUDA:", torch.cuda.is_available())
print("MPS:", torch.backends.mps.is_available())
print("Any accelerator:", torch.accelerator.is_available())

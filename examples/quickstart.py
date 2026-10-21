""" Quickstart: this demonstrates basic DataLoader usage."""

import numpy as np
from dataloader import DataLoader

# lets make up a dataset of 10 samples, where each of them is a 4-element array
dataset = [np.array([i, i+1, i+2, i+3]) for i in range(10)]
    # something like [[0, 1, 2, 3], [1, 2, 3, 4] ...] will be created
    
print(">>> Default (no shuffle, no drop_last) <<<")
loader = DataLoader(dataset, batch_size=3)
for batch in loader:
    print(batch)
    
print(">>> Set drop_last as True) <<<")
loader = DataLoader(dataset, batch_size=3, drop_last=True)
for batch in loader:
    print(batch)
    

print(">>> Set shuffle=True, and seed=42 <<<")
loader = DataLoader(dataset, batch_size=3, shuffle=True, seed=42)
for batch in loader:
    print(batch)
import numpy as np
import pytest
from dataloader import DataLoader

def test_len_no_drop_last():
    loader = DataLoader(list(range(10)), batch_size=3)
    assert len(loader) == 4

def test_len_drop_last():
    loader = DataLoader(list(range(10)), batch_size=3, drop_last=True)
    assert len(loader) == 3
    
def test_batch_shape():
    dataset = [np.ones(4) for _ in range(9)]
    loader = DataLoader(dataset, batch_size=3)
    batches = list(loader)
    assert batches[0].shape == (3, 4)
    
def test_drop_last_removes_final_batch():
    loader = DataLoader(list(range(10)), batch_size=3, drop_last=True)
    batches = list(loader)
    assert len(batches) == 3
    assert all(len(b) == 3 for b in batches)
    
def test_shuffle_reproducible():
    dataset = list(range(20))
    loader1 = DataLoader(dataset, batch_size=4, shuffle=True, seed=4)
    loader2 = DataLoader(dataset, batch_size=4, shuffle=True, seed=4)
    for b1, b2 in zip(loader1, loader2):
        assert np.array_equal(b1, b2)
        
def test_invalid_batch_size():
    with pytest.raises(ValueError):
        DataLoader(list(range(10)), batch_size=0)
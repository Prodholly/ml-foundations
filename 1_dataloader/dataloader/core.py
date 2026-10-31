import math
import random
from collections.abc import Iterator, Sequence
from typing import Generic, TypeVar, Optional, Callable

import numpy as np

T = TypeVar("T")

class DataLoader(Generic[T]):
    """this is a typed DataLoader for iterating over datasets in batches.

    Args:
        dataset: Any sequence of samples
        batch_size: Number of samples per batch.
        shuffle: If True, reshuffle at the start of each epoch
        drop_last: If True, drop the final batch if smaller than batch_size
        collate_fn: Function to merge samples into a batch. Defaults to np.stack
        seed: Optional random seed for reproducibility
    """
    def __init__(self, dataset: Sequence[T], batch_size: int, shuffle: bool = False,
                 drop_last: bool = False, collate_fn: Optional[Callable[[list[T]], np.ndarray]] = None,
                 seed: Optional[int] = None) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if len(dataset) == 0:
            raise ValueError("dataset cannot be empty")
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn if collate_fn is not None else np.stack
        self._rng = random.Random(seed)
    
    def __len__(self) -> int:
        """number of batches per epoch"""
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)
    
    def __iter__(self) -> Iterator[np.ndarray]:
        """yield one collated batch per iteration"""
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            self._rng.shuffle(indices) # shuffle is a built-in method on random.Random | functions to make shuffle reproducible
            
        """ we want to pick the batch of 3 numbers, right?"""
        for start in range(0, len(self.dataset), self.batch_size): # This steps through the index positions in jumps of batch_size. For 10 items, batch_size=3: start takes values 0, 3, 6, 9
            batch_indices = indices[start: start + self.batch_size] # Slices out the next chunk of indices. At start=6: batch_indices = [6, 7, 8]. At start=9: batch_indices = [9]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break
            samples = [self.dataset[i] for i in batch_indices]
            yield self.collate_fn(samples)
        
    
        
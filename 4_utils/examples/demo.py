"""Demo: this shows the three utilities on the kind of jobs they do in an ML pipeline."""

import numpy as np

from utils import batch_items, check_shape, safe_parse_int

print(">>> batch_items: split 10 sample IDs into batches of 4 <<<")
sample_ids = list(range(10))
for batch in batch_items(sample_ids, 4):
    print(batch)
    # the last batch is smaller: 10 isn't a multiple of 4


print(">>> check_shape: confirm a batch is (batch, features) before it reaches a model <<<")
batch = np.random.default_rng(0).random((4, 3))
print("matches (4, 3)?", check_shape(batch, (4, 3)))

try:
    check_shape(batch.T, (4, 3))  # a transposed batch, a common mistake
except ValueError as error:
    print("caught:", error)


print(">>> safe_parse_int: clean a column of values read from a file <<<")
raw_values = ["32", "abc", "", None, 3.9, " 7 "]
for value in raw_values:
    print(f"{value!r:>6} -> {safe_parse_int(value, default=-1)}")

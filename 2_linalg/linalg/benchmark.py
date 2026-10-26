import numpy as np
import time

# Original loop-based matmul
from linalg.ops import matmul


# now lets create the benchmark
if __name__ == "__main__":
    # i set up large matrices so that we can see meaningful differences in the timing
    A = np.random.rand(500, 500)
    B = np.random.rand(500, 500)

    # my loop matmul
    start = time.time()
    matmul(A, B)
    my_time = time.time() - start
    print(f"Loop matmul:   {my_time:.4f} seconds")

    # NumPy matmul
    start = time.time()
    np.matmul(A, B)
    numpy_time = time.time() - start
    print(f"NumPy matmul:  {numpy_time:.4f} seconds")

    # Einsum matmul
    start = time.time()
    np.einsum('ik,kj->ij', A, B)
    einsum_time = time.time() - start
    print(f"einsum matmul: {einsum_time:.4f} seconds")

    print(f"\nNumPy is {my_time / numpy_time:.0f}x faster than my matmul loop")
    print(f"NumPy is {einsum_time / numpy_time:.1f}x faster than einsum")

# the results
# Loop matmul:   39.9447 seconds
# NumPy matmul:  0.0027 seconds
# einsum matmul: 0.0204 seconds

# NumPy is 14659x faster than my matmul loop
# NumPy is 7.5x faster than einsum

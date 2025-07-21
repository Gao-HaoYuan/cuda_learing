import numpy as np
import cuda_add

a = np.ones(1024, dtype=np.float32)
b = np.ones(1024, dtype=np.float32) * 2

out = cuda_add.add(a, b)
print(out[:7])  # [3. 3. 3. 3. 3.]

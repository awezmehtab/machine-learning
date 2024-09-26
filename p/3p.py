import numpy as np

x = np.arange(8).reshape(2,4)
print(np.array_split(x, 2, axis=0))
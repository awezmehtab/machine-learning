import numpy as np

import matplotlib.pyplot as plt

# Your array of positive integers
array = np.array([0,0,0,2,2,2,4,4,4,2,2,2,0,0,0])

# Reshape the array into a 2D array with one row
array = np.reshape(array, (1, -1))

# Display the grayscale image
plt.imshow(array, cmap='gray', vmin=0, vmax=4)
plt.show()

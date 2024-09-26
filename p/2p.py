import numpy as np

x = np.array([[-3,2],[-1,1],[-1,-1],[2,2],[1,-1]])
labels = np.array([1,-1,-1,-1,-1], ndmin=2).T
times_mistaken = np.array([2,4,2,1,0], ndmin=2).T

print(np.sum(x*labels*times_mistaken, axis=0))

print(np.sum(labels*times_mistaken, axis=0))
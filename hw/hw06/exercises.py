import numpy as np
import sympy as smp

X = np.array([[0,1,2],[0,1,2]])
Y = np.array([[0,1,0]])

X_new = np.array([[0,1,1],[1,1,0]])
X_new = np.concatenate((np.array([1,1,1], ndmin=2), X_new), axis=0)
w = smp.symbols('w1 w2 w3')
w = np.array(w).reshape(-1, 1)

result = np.dot(w.T, X_new)
print(result)
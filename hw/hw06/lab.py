import numpy as np

def softmax(x):
    """Computes the softmax function for x which is a array"""
    return np.exp(x) / np.sum(np.exp(x))

def NLL(a, y):
    return -np.sum(y * np.log(a))

def w_grad(w, x, y):
    a = softmax(w.T @ x)
    return x @ (a.T - y.T)

def layer_output(w, w0, f, x):
    """ w is weight matrix, w0 is a vector which contains all constants.
    f is function which is applied on pre-activated vector output.
    x is input vector """

    return f(w.T @ x + w0)

w = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
w0 = np.array([[-1, -1, -1, -1]]).T
f = lambda x: np.maximum(x, 0)

w2 = np.array([[1, -1]] * 4)
w20 = np.array([[0, 2]]).T

output_1 = layer_output(w, w0, f, np.array([[3, 14]]).T)
output_2 = layer_output(w2, w20, lambda x: softmax(x), output_1)
print(output_2.tolist())
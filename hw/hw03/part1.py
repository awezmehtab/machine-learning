import numpy as np

# QUESTION 1

# th = np.array([[0, 1, -0.5]]).T
# data = np.array([[200, 800, 200, 800], [0.2,  0.2,  0.8,  0.8], [1,1,1,1]])
# y = np.array([[-1,-1,1,1]])
# data_new = data*(np.concatenate(([np.ones((1,4))*0.001, np.ones((2,4))])))
# print(data_new/data)

# margins = y * (th.T @ data_new) / np.linalg.norm(th)
# R2s = np.ones((1,3)) @ data_new**2 

# # print(margins, R2s)
# print(no_mistakes(data_new, y))

def no_mistakes(data, labels):
    d, n = data.shape
    th = np.zeros((d,1))
    mistakes = 0
    while True:
        mila = False
        for i in range(n):
            x_i = data[:, i].reshape((d,1))
            y_i = labels[:, i].reshape((1,1))
            if y_i * (th.T @ x_i) <= 0:
                th = th + y_i * x_i
                mistakes += 1
                mila = True
        if not mila:
            break

    return mistakes

def perceptron(data, labels, T=10000):
    d, n = data.shape
    th = np.zeros((d,1))
    th0 = np.zeros((1,1))
    mistakes = 0
    for i in range(T):
        mila = False
        for i in range(n):
            x_i = data[:, i].reshape((d,1))
            y_i = labels[:, i].reshape((1,1))
            if y_i * (th.T @ x_i + th0) <= 0:
                th = th + y_i * x_i
                th0 = th0 + y_i
                mistakes += 1
                mila = True
        if not mila:
            break

    return (th, th0, mistakes)

def classifier(perceptron, data):
    th, th0, mistakes = perceptron
    x = np.sign(th.T @ data + th0)
    x[x == 0] = -1
    return x

def one_hot_internal(x,k):
    v = np.zeros(k)
    v[x-1] = 1
    return v

def margin(perceptron, data, labels):
    th, th0, mistakes = perceptron
    margins = labels * (th.T @ data + th0) / np.linalg.norm(th)
    return margins

data = np.array([[2, 3,  4,  5]])
data_hot = np.array([one_hot_internal(x,6) for x in data[0,:]]).T
all_data_hot = np.array([one_hot_internal(x,6) for x in range(1,7)]).T
labels = np.array([[1, 1, -1, -1, 1, 1]])

print(perceptron(all_data_hot, labels))
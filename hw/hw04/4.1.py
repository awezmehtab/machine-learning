import numpy as np

def margins(data, th, th0, labels):
    return (th.T @ data + th0)*labels/np.linalg.norm(th)

def hinge_losses(data, th, th0, labels, ref):
    margins_data = margins(data, th, th0, labels)
    margins_ref = margins_data <= ref
    return (1 - margins_data/ref)*margins_ref

def objective(data, th, th0, labels, ref, lam):
    d, n = data.shape
    return np.sum(hinge_losses(data, th, th0, labels, ref))/n + lam/(ref**2)

def svm_objective(data, labels, th, th0, lam):
    length_th = np.linalg.norm(th)
    return objective(data, th, th0, labels, 1/length_th, lam)

data = np.array([[1,1,1.5,2,2,3,3], [1,3,1.5,1,3,1,3]])
labels = np.array([[1,-1,-1,1,-1,1,-1]])

th1 = np.array([[0.01280916], [-1.42043497]])
th01 = np.array([[2.387955038624247]])

th2 = np.array([[0.45589866], [-4.50220738]])
th02 = np.array([[5.056803830699447]])

th3 = np.array([[0.04828952], [-4.13159675]])
th03 = np.array([[5.110597526206023]])

lambdas = [0, 0.001, 0.03]

for lamda in lambdas:
    for i in range(1,4):
        print(f'svm_objective for th{i} and lamda {lamda} ',svm_objective(data, labels, globals()['th'+str(i)], globals()['th0'+str(i)], lamda))
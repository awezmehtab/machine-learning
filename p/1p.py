import numpy as np

x = np.array([[0,0,0],[0,1,0],[1,0,0],[0,0,1],[0,1,1],[1,0,1],[1,1,0],[1,1,1]])

T = 1000
th = np.array([0,0,0])
th0 = 0
labels = np.array([-1,-1,-1,-1,-1,-1,-1,1])

mistakes = 0

for j in range(T):
    mila = False
    for i in range(8):
        if labels[i]*(np.dot(x[i],th)+th0) <= 0:
            th = th + labels[i]*x[i]
            th0 = th0 + labels[i]
            mila = True
            mistakes += 1
    if not mila:
        print(j, ' iterations pe stopped')
        break

print('Total mistakes: ', mistakes)
print('Final theta: ', th)
print('Final theta0: ', th0)
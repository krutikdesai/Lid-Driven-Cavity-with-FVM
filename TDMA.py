import numpy as np


def tdma(Aw, Ap, Ae, b):

    n = len(b)
    phi = np.empty(n, dtype=float)

    for i in range(1, n):
        Ap[i] -= Aw[i]*Ae[i-1]/Ap[i-1]
        b[i] -= Aw[i]*b[i-1]/Ap[i-1]

    phi[-1] = b[-1]/Ap[-1]

    for j in range(n-2,-1,-1):
        phi[j] = (b[j] - Ae[j]*phi[j+1])/Ap[j]
        
    return phi


Aw = np.array([0,3,2,1], dtype=float)
Ap = np.array([1,4,3,3], dtype=float)
Ae = np.array([4,1,4,0], dtype=float)
b = np.array([2,1,2,5], dtype=float)

print(tdma(Aw,Ap,Ae,b))

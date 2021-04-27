import numpy as np
import pandas as pd

itr_limit = 1000
conv_crit = 0.0000001


def stop_cond(A, b, x):
    n = len(b)
    res = (A @ x - b)
    magnitude = (res @ res) ** 0.5

    if magnitude < conv_crit:
        return True
    else:
        return False


def GS(A, b, N=-1, x=None):
    n = len(b)
    itr = 0
    converged = False
    if x is None:
        x = np.zeros(n)

    while not converged and itr < itr_limit and itr != N:

        for i in range(n):
            dot = 0
            for j in range(n):
                if (i != j):
                    dot += A[i, j] * x[j]

            x[i] = (b[i] - dot) / A[i, i]

        itr += 1
        converged = stop_cond(A, b, x)

    if not converged:
        print("Iteration limit reached without convergence.")

    return x

Ni = 10
Nj = 10
dx = 0.1  # in m
dy = 0.1  # in m
K = 384.1  # Conductivity in W/mK
Su = 0  # Source in W/m^3
Sp = 0  # Source in W/(K.m^3)

Tw = 2500  # Temperatures in K
Ts = 400
Tn = 300
Te = 200
qw = 0  # Fluxes in W/m^2
qs = 0
qn = 0
qe = 0


def pad(x, i, j):
    if j < 0 or j >= x.shape[1]:
        return 0
    else:
        return x[i,j]


def diffusion_steady():
    n = Ni * Nj
    A = np.zeros((n,n))
    b = np.full(n, Su * dx * dy, dtype="double")

    for i in range(n):
        if i < Nj:
            A[i,i] += 2 * K * (dy / dx)
            b[i] += 2 * K * Tw * (dy / dx) + (qw * dy)
        else:
            A[i, i-Nj] -= K * dy / dx

        if i >= n - Nj:
            A[i,i] += 2 * K * (dy / dx)
            b[i] += 2 * K * Te * (dy / dx) + (qe * dy)
        else:
            A[i, i+Nj] -= K * dy / dx

        if i % Nj == 0: 
            if i != 0 and i != (n-Nj):
                A[i,i] += 2 * K * (dx / dy)
                b[i] += 2 * K * Ts * (dx / dy) + (qs * dx)
        else:
            A[i, i-1] -= K * dx / dy

        if (i + 1) % Nj == 0: 
            if i != (Nj-1) and i != (n-1):
                A[i,i] += 2 * K * (dx / dy)
                b[i] += 2 * K * Tn * (dx / dy) + (qn * dx)
        else:
            A[i, i+1] -= K * dx / dy

        A[i,i] -= pad(A,i,i-Nj) + pad(A,i,i-1) + pad(A,i,i+1) + pad(A,i,i+Nj) + Sp * dx * dy
    df1 = pd.DataFrame(A)
    df2 = pd.DataFrame(b)
    with pd.ExcelWriter('data.xlsx') as writer:
        df1.to_excel(writer, sheet_name='Sheet1')
        df2.to_excel(writer, sheet_name='Sheet2')
    return GS(A,b)


print(diffusion_steady())


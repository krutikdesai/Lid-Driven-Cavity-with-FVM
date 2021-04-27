import numpy as np
import pandas as pd

itr_limit = 20
conv_crit = 0.001


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
                if i != j:
                    dot += A[i, j] * x[j]

            x[i] = (b[i] - dot) / A[i, i]

        itr += 1
        converged = stop_cond(A, b, x)

    if not converged:
        print("Iteration limit reached without convergence.")

    return x


df = pd.read_excel('data.xlsx', engine='openpyxl')
A = df.to_numpy()
b = np.array([3, 7, 11, 16, 21, 26, 24, 28, 23], dtype="double")
print(GS(A, b))

import numpy as np
import pandas as pd


def swap(A, b, i, j):
    A[j, :], A[i, :] = A[i, :], A[j, :].copy()
    b[j], b[i] = b[i], b[j].copy()
    return


def forward(A, b, n):

    for i in range(0, n - 1):
        if A[i][i] == 0:
            pivot = np.nonzero(A[i:, i])[0][0]
            swap(A, b, i, pivot)
        else:
            for j in range(i + 1, n):
                d = A[j][i] / A[i][i]
                for k in range(i + 1, n):
                    A[j, k] = A[j, k] - (d * A[i, k])
                b[j] = b[j] - (d * b[i])
    return


def back(A, b, n):

    if A[n-1, n-1] == 0:
        raise ValueError("System has infinite solutions.")

    for i in range(n-1, -1, -1):
        b[i] = b[i]/A[i, i]
        for j in range(i-1, -1, -1):
            b[j] -= A[j, i]*b[i]

    return


def gaussian_elimination(A, b):
    n = len(b)
    forward(A, b, n)
    back(A, b, n)
    return b


df = pd.read_excel('data.xlsx', engine='openpyxl')
A = df.to_numpy()
b = np.array([3, 7, 11, 16, 21, 26, 24, 28, 23], dtype="float")
print(gaussian_elimination(A, b))
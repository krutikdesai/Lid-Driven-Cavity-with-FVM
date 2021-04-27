import numpy as np
import pandas as pd

conv_crit = 0.000001
alpha = 1
limit = 500

L = 1        # in m
Ni = 10
Nj = 10
n = Ni * Nj
dx = L/Ni    # in m
dy = L/Nj    # in m
K = 384.1    # Conductivity in W/mK

Tw = 900     # Boundary temperatures in K
Ts = 288
Tn = 288
Te = 288
qw = 0       # Boundary fluxes in W/m^2
qs = 0
qn = 0
qe = 0

def pad(x, i):
    if i < 0 or i >= x.shape[0]:
        return 0
    else:
        return x[i]


def SIP(Aw, As, Ap, An, Ae, b):

    Lw = np.zeros(n, dtype="double")
    Ls = np.zeros(n, dtype="double")
    Lp = np.zeros(n, dtype="double")
    Un = np.zeros(n, dtype="double")
    Ue = np.zeros(n, dtype="double")

    for i in range(n):
        Lw[i] = Aw[i] / (1 + alpha * Un[i - Nj])
        Ls[i] = As[i] / (1 + alpha * Ue[i - 1])
        Lp[i] = Ap[i] + alpha * (Lw[i] * Un[i - Nj] + Ls[i] * Ue[i - 1]) - Lw[i] * Ue[i - Nj] - Ls[i] * Un[i - 1]
        Un[i] = (An[i] - alpha * Lw[i] * Un[i - Nj]) / Lp[i]
        Ue[i] = (Ae[i] - alpha * Ls[i] * Ue[i - 1]) / Lp[i]

    x = np.zeros(n, dtype="double")
    R = np.empty(n, dtype="double")
    delta = np.empty(n, dtype="double")
    error = 1
    itr = 0

    while error > conv_crit and itr <= limit:

        for k in range(n):
            rho = b[k] - Aw[k] * pad(x, k - Nj) - As[k] * pad(x, k - 1) \
                  - Ae[k] * pad(x, k + Nj) - An[k] * pad(x, k + 1) \
                  - Ap[k] * pad(x, k)
            R[k] = (rho - Ls[k] * pad(R, k - 1) - Lw[k] * pad(R, k - Nj)) / Lp[k]

        for j in range(n - 1, -1, -1):
            delta[j] = R[j] - Un[j] * pad(delta, j + 1) - Ue[j] * pad(delta, j + Nj)

        error = (delta @ delta) ** 0.5
        x += delta
        itr += 1

    return x


def diffusion_unsteady():

    Dx = K * (dy / dx)
    Dy = K * (dx / dy)
    Su = np.zeros(n, dtype="double")   # Source in W/m^3
    Sp = np.zeros(n, dtype="double")   # Source in W/(K.m^3)
    Su[55] = 50000000

    T = np.full((n,1), 288, dtype="double")  # Initial Conditions
    Apo = 2 * (Dx + Dy)

    for j in range(1, 50):

        Ap = np.zeros(n, dtype="double")
        b  = Apo * T[:, j - 1]

        Aw = np.full(n, -Dx, dtype="double")
        As = np.full(n, -Dy, dtype="double")
        Ae = Aw.copy()
        An = As.copy()

        for i in range(n):
            if i < Nj:
                Aw[i] = 0
                Ap[i] += 2 * Dx
                b[i] += 2 * Dx * Tw + (qw * dy)

            if i >= n - Nj:
                Ae[i] = 0
                Ap[i] += 2 * Dx
                b[i] += 2 * Dx * Te + (qe * dy)

            if i % Nj == 0:
                As[i] = 0
                if i != 0 and i != (n - Nj):
                    Ap[i] += 2 * Dy
                    b[i] += 2 * Dy * Ts + (qs * dx)

            if (i + 1) % Nj == 0:
                An[i] = 0
                if i != (Nj - 1) and i != (n - 1):
                    Ap[i] += 2 * Dy
                    b[i] += 2 * Dy * Tn + (qn * dx)

            Ap[i] -= Aw[i] + As[i] + An[i] + Ae[i] + (Sp[i] * dx * dy) - Apo
            b[i] += Su[i] * dx * dy

        T = np.insert(T, j, SIP(Aw, As, Ap, An, Ae, b), axis=1)

    return T


df = pd.DataFrame(diffusion_unsteady())
df.to_excel('data.xlsx', sheet_name='Sheet1')

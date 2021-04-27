import numpy as np
import pandas as pd

conv_crit = 0.0000001
alpha = 1
limit = 500

L = 1      # in m
Ni = 25
Nj = 25
n = Ni * Nj
dx = L/Ni  # in m
dy = L/Ni  # in m
K = 0.04   # Conductivity in W/mK
C = 1297   # Volumetric heat capacity in J/(K.m^3)

Tw = 1000  # Temperatures in K
Ts = 288
Tn = 288
Te = 500
qw = 0     # Fluxes in W/m^2
qs = 0
qn = 0
qe = 0


def pad(x, i):
    if i < 0 or i >= x.shape[0]:
        return 0
    else:
        return x[i]


def scheme(Pe):
    return max(0, (1 - 0.1*abs(Pe))**5)


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


def convdiff_steady():

    Dx = K * (dy / dx)
    Dy = K * (dx / dy)
    Su = np.zeros(n, dtype="double")   # Source in W/m^3
    Sp = np.zeros(n, dtype="double")   # Source in W/(K.m^3)

    Ap = np.zeros(n, dtype="double")
    b  = np.zeros(n, dtype="double")
    Fx = np.full(n + Nj, C * 0.004 * dy, dtype="double") # Velocity data in x direction
    Fy = np.full(n + Ni, C * 0 * dx, dtype="double")     # Velocity data in y direction

    Aw = np.empty(n, dtype="double")
    As = np.empty(n, dtype="double")
    Ae = np.empty(n, dtype="double")
    An = np.empty(n, dtype="double")

    for i in range(n):

        Aw[i] = -Dx * scheme(Fx[i]/Dx)                 - max(0, Fx[i])
        Ae[i] = -Dx * scheme(Fx[i+Nj]/Dx)              - max(0,-Fx[i+Nj])
        As[i] = -Dy * scheme(Fy[i + int(i/Nj)]/Dy)     - max(0, Fy[i + int(i/Nj)])
        An[i] = -Dy * scheme(Fy[i + int(i/Nj) + 1]/Dy) - max(0,-Fy[i + int(i/Nj) + 1])

        if i < Nj:
            Aw[i] = 0
            tmp = 2 * Dx + max(0, Fx[i])
            Ap[i] += tmp
            b[i]  += tmp * Tw + (qw * dy)
        
        elif i >= n - Nj:
            Ae[i] = 0
            tmp = 2 * Dx + max(0, -Fx[i+Nj])
            Ap[i] += tmp
            b[i]  += tmp * Te + (qe * dy)
        
        if i % Nj == 0:
            As[i] = 0
            if i != 0 and i != n-Nj:
                tmp = 2 * Dy + max(0, Fy[i + int(i/Nj)])
                Ap[i] += tmp
                b[i]  += tmp * Ts + (qs * dx)
        
        elif (i + 1) % Nj == 0:
            An[i] = 0
            if i != Nj-1 and i != n-1:
                tmp = 2 * Dy + max(0, -Fy[i + int(i/Nj) + 1])
                Ap[i] += tmp
                b[i]  += tmp * Tn + (qn * dx)

        Ap[i] -= Aw[i] + As[i] + An[i] + Ae[i] + (Sp[i] * dx * dy)
        b[i] += Su[i] * dx * dy

    return SIP(Aw, As, Ap, An, Ae, b)


T = convdiff_steady()
df = pd.DataFrame(T)
df.to_excel('data.xlsx', sheet_name='Sheet3')





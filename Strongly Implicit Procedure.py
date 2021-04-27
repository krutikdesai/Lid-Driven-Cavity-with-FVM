import numpy as np

conv_crit = 0.0000001
alpha = 1
limit = 500


def pad(x, i):
    if i < 0 or i >= x.shape[0]:
        return 0
    else:
        return x[i]


def SIP(Aw, As, Ap, An, Ae, b, Ni, Nj):

    n = Ni*Nj
    Lw = np.empty(n, dtype="double")
    Ls = np.empty(n, dtype="double")
    Lp = np.empty(n, dtype="double")
    Un = np.empty(n, dtype="double")
    Ue = np.empty(n, dtype="double")

    for i in range(n):
        Lw[i] = Aw[i] / (1 + alpha * pad(Un, i-Nj))
        Ls[i] = As[i] / (1 + alpha * pad(Ue, i-1))
        Lp[i] = Ap[i] + alpha * (Lw[i] * pad(Un, i-Nj) + Ls[i] * pad(Ue, i-1)) - Lw[i] * pad(Ue, i-Nj) - Ls[i] * pad(Un, i-1)
        Un[i] = (An[i] - alpha * Lw[i] * pad(Un, i-Nj)) / Lp[i]
        Ue[i] = (Ae[i] - alpha * Ls[i] * pad(Ue, i-1)) / Lp[i]

    x = np.zeros(n, dtype="double")
    R = np.empty(n, dtype="double")
    delta = np.empty(n, dtype="double")
    error = 1
    itr = 0

    while error > conv_crit and itr <= limit:

        for k in range(n):
            rho = b[k] - Aw[k] * pad(x, k - Nj) - As[k] * pad(x, k - 1) - Ae[k] * pad(x, k + Nj) - An[k] * pad(x, k + 1) - Ap[k] * pad(x, k)
            R[k] = (rho - Ls[k] * pad(R, k - 1) - Lw[k] * pad(R, k - Nj)) / Lp[k]

        for j in range(n - 1, -1, -1):
            delta[j] = R[j] - Un[j] * pad(delta, j + 1) - Ue[j] * pad(delta, j + Nj)

        error = (delta @ delta) ** 0.5
        x += delta
        print(R)
        itr += 1

    return x


Aw = np.array([0,0,1,2], dtype="double")
As = np.array([0,1,2,3], dtype="double")
Ap = np.array([1,2,3,4], dtype="double")
An = np.array([1,2,3,0], dtype="double")
Ae = np.array([1,2,0,0], dtype="double")
b = np.array([3,7,9,9], dtype="double")

# Aw = np.array([0,0,0,1,2,3,4,5,6], dtype="double")
# As = np.array([0,1,2,3,4,5,6,7,8], dtype="double")
# Ap = np.array([1,2,3,4,5,6,7,8,9], dtype="double")
# An = np.array([1,2,3,4,5,6,7,8,0], dtype="double")
# Ae = np.array([1,2,3,4,5,6,0,0,0], dtype="double")
# b = np.array([3,7,11,16,21,26,24,28,23], dtype="double")

print(SIP(Aw, As, Ap, An, Ae, b, 2, 2))

# ....Nw  = np.empty(n, dtype="double")
# ....Nnw = np.empty(n, dtype="double")
# ....Ns  = np.empty(n, dtype="double")
# ....Np  = np.empty(n, dtype="double")
# ....Nn  = np.empty(n, dtype="double")
# ....Nse = np.empty(n, dtype="double")
# ....Ne  = np.empty(n, dtype="double")

# ....for j in range(n):
# ........Nw[j]  = Lw[j] - Aw[j]
# ........Nnw[j] = Lw[j]*Un[j-Nj]
# ........Ns[j]  = Ls[j] - As[j]
# ........Np[j]  = Lw[j]*Ue[j-Nj] + Ls[j]*Un[j-1] + Lp[j] - Ap[j]
# ........Nn[j]  = Un[j]*Lp[j] - An[j]
# ........Nse[j] = Ls[j]*Ue[j-1]
# ........Ne[j]  = Ue[j]*Lp[j] - Ae[j]

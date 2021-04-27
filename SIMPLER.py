import numpy as np
import pandas as pd

conv_crit = 0.000001
alpha = 0.9
limit = 500

L = 1         # in m
Ni = 30
Nj = 30
n = Ni * Nj
dx = L/Ni     # in m
dy = L/Nj     # in m
mu = 0.01     # Viscosity in Pa.s
rho = 100     # Density in kg/m^3
Re = 100	  # Reynolds Number

w = 1       # Under-relaxation for velocity
w_p = 0.8     # Under-relaxation for pressure
uw = 0        # Wall velocities in m/s (BCs)
us = 0
ue = 0
un = mu*Re*Nj/rho
vw = 0
vs = 0
ve = 0
vn = 0


def pad(x, i):

    if i < 0 or i >= x.shape[0]:
        return 0
    else:
        return x[i]


def boundary(j, p, q):

    wall = set()
    if j < p:
        wall.update({"W", "V"})
    elif j >= p*(q-1):
        wall.update({"E", "V"})
    if j % p == 0:
        wall.update({"S", "H"})
    elif (j+1) % p == 0:
        wall.update({"N", "H"})
    return wall


def export(U, V, P):

    Uc = np.empty((Nj, Ni))
    Vc = np.empty((Nj, Ni))
    Pc = np.empty((Nj, Ni))

    for j in range(Ni):
        for i in range(Nj):
            k = i + j * Nj
            Uc[i][j] = 0.5 * (U[k] + U[k + Nj])
            Vc[i][j] = 0.5 * (V[k + int(k/Nj)] + V[k + int(k/Nj) + 1])
            Pc[i][j] = P[k]

    writer = pd.ExcelWriter('solution.xlsx', engine='openpyxl')

    pd.DataFrame(Uc).to_excel(writer, sheet_name='Udata')
    pd.DataFrame(Vc).to_excel(writer, sheet_name='Vdata')
    pd.DataFrame(Pc).to_excel(writer, sheet_name='Pdata')
    writer.save()
    return


def SIP(Aw, As, Ap, An, Ae, b, length, lim=limit):

    Lw = np.empty(length, dtype="float")
    Ls = np.empty(length, dtype="float")
    Lp = np.empty(length, dtype="float")
    Un = np.empty(length, dtype="float")
    Ue = np.empty(length, dtype="float")

    for i in range(length):
        Lw[i] = Aw[i] / (1 + alpha * Un[i - Nj])
        Ls[i] = As[i] / (1 + alpha * Ue[i - 1])
        Lp[i] = Ap[i] + alpha * (Lw[i] * Un[i - Nj] + Ls[i] * Ue[i - 1]) - Lw[i] * Ue[i - Nj] - Ls[i] * Un[i - 1]
        Un[i] = (An[i] - alpha * Lw[i] * Un[i - Nj]) / Lp[i]
        Ue[i] = (Ae[i] - alpha * Ls[i] * Ue[i - 1]) / Lp[i]

    x = np.zeros(length, dtype="float")
    R = np.empty(length, dtype="float")
    delta = np.empty(length, dtype="float")
    error = 1
    itr = 0

    while error > conv_crit and itr < lim:

        error = 0
        for k in range(length):
            rho = b[k] - Aw[k] * pad(x, k - Nj) - As[k] * pad(x, k - 1) \
                  - Ae[k] * pad(x, k + Nj) - An[k] * pad(x, k + 1) \
                  - Ap[k] * pad(x, k)
            R[k] = (rho - Ls[k] * pad(R, k - 1) - Lw[k] * pad(R, k - Nj)) / Lp[k]

        for j in range(length - 1, -1, -1):
            delta[j] = R[j] - Un[j] * pad(delta, j + 1) - Ue[j] * pad(delta, j + Nj)
            error += delta[j] ** 2

        error = error ** 0.5
        x += delta
        itr += 1
    
    print(".........", itr)
    return x


def NS_steady():

    Dx = mu * (dy / dx)
    Dy = mu * (dx / dy)
    err_src = 1
    itr = 0

    U = np.zeros(n + Nj, dtype="float")    # X-Velocity data u* at cell faces
    V = np.zeros(n + Ni, dtype="float")    # Y-Velocity data v* at cell faces
    P = np.zeros(n, dtype="float")         # Pressure data p* at cell centres

    Ap = np.empty(n+Nj+Ni, dtype="float")
    b  = np.empty(n+Nj+Ni, dtype="float")
    Aw = np.empty(n+Nj+Ni, dtype="float")
    As = np.empty(n+Nj+Ni, dtype="float")
    Ae = np.empty(n+Nj+Ni, dtype="float")
    An = np.empty(n+Nj+Ni, dtype="float")
    de = np.empty(n+Nj+Ni, dtype="float")
    dn = np.empty(n+Ni+Ni, dtype="float")

    while err_src > 0.000001 and itr < 500:

        Ap.fill(0)
        b.fill(0)
        U_H = np.empty(n + Nj, dtype="float")
        V_H = np.empty(n + Ni, dtype="float")
        U_S = np.empty(n + Nj, dtype="float")
        V_S = np.empty(n + Ni, dtype="float")

        for i in range(n+Nj):

            walls = boundary(i, Nj, Ni+1)

            if "V" in walls:
                Ap[i] = 1
                Aw[i] = Ae[i] = As[i] = An[i] = 0
                b[i] = uw if "W" in walls else ue
            else:
                Fw = rho * dy * 0.5 * (U[i] + U[i-Nj])
                Fe = rho * dy * 0.5 * (U[i] + U[i+Nj])
                Aw[i] = -Dx - max(0, Fw)
                Ae[i] = -Dx - max(0,-Fe)
                As[i] = An[i] = 0

                if "S" in walls:
                    tmp = (8/3) * Dy + max(0, rho*dx*vs)
                    Ap[i] += tmp
                    b[i] += tmp * us
                    An[i] -= Dy/3
                else:
                    Fs = rho * dx * 0.5 * (V[i + int(i/Nj)] + V[i + int(i/Nj) - Nj - 1])
                    As[i] -= Dy + max(0, Fs)

                if "N" in walls:
                    tmp = (8/3) * Dy + max(0, -rho*dx*vn)
                    Ap[i] += tmp
                    b[i] += tmp * un
                    As[i] -= Dy/3
                else:
                    Fn = rho * dx * 0.5 * (V[i + int(i/Nj) + 1] + V[i + int(i/Nj) - Nj])
                    An[i] -= Dy + max(0,-Fn)

                Ap[i] -= Aw[i] + As[i] + An[i] + Ae[i]
                #b[i]  += dy * (P[i-Nj] - P[i])
            
            U_H[i] = (b[i] - (Aw[i] * pad(U, i-Nj) + As[i] * pad(U, i-1) + An[i] * pad(U, i+1) + Ae[i] * pad(U, i+Nj)))/Ap[i]
            de[i] = dy/Ap[i]

        Ap.fill(0)
        b.fill(0)

        for i in range(n+Ni):

            walls = boundary(i, Nj+1, Ni)

            if "H" in walls:
                Ap[i] = 1
                Aw[i] = Ae[i] = As[i] = An[i] = 0
                b[i] = vs if "S" in walls else vn
            else:
                Fs = rho * dx * 0.5 * (V[i] + V[i - 1])
                Fn = rho * dx * 0.5 * (V[i] + V[i + 1])
                As[i] = -Dy - max(0, Fs)
                An[i] = -Dy - max(0,-Fn)
                Aw[i] = Ae[i] = 0

                if "W" in walls:
                    tmp = (8/3) * Dx + max(0, rho*dy*uw)
                    Ap[i] += tmp
                    b[i] += tmp * vw
                    Ae[i] -= Dx/3
                else:
                    Fw = rho * dx * 0.5 * (U[i - int(i/(Nj+1))] + U[i - int(i/(Nj+1)) - 1])
                    Aw[i] -= Dx + max(0, Fw)

                if "E" in walls:
                    tmp = (8/3) * Dx + max(0, rho*dy*ue)
                    Ap[i] += tmp
                    b[i] += tmp * ve
                    Ae[i] -= Dx/3
                else:
                    Fe = rho * dx * 0.5 * (U[i - int(i/(Nj+1)) + Nj] + U[i - int(i/(Nj+1)) + Nj - 1])
                    Ae[i] -= Dx + max(0,-Fe)

                Ap[i] -= Aw[i] + As[i] + An[i] + Ae[i]
                #b[i] += dx * (P[i - int(i/(Nj+1)) - 1] - P[i - int(i/(Nj+1))])
            
            V_H[i] = (b[i] - (Aw[i] * pad(V, i-1-Nj) + As[i] * pad(V, i-1) + An[i] * pad(V, i+1) + Ae[i] * pad(V, i+1+Nj)))/Ap[i]
            dn[i] = dx/Ap[i]

        U = U_H
        V = V_H

        for k in range(n):

            walls = boundary(k, Nj, Ni)

            Aw[k] = 0 if "W" in walls else -rho * dy * de[k]
            Ae[k] = 0 if "E" in walls else -rho * dy * de[k+Nj]
            As[k] = 0 if "S" in walls else -rho * dx * dn[k + int(k/Nj)]
            An[k] = 0 if "N" in walls else -rho * dx * dn[k + int(k/Nj) + 1]

            Ap[k] = -(Aw[k] + As[k] + An[k] + Ae[k])
            b[k]  = rho * dy * (U[k] - U[k+Nj]) + rho * dx * (V[k + int(k/Nj)] - V[k + int(k/Nj) + 1])

        P = SIP(Aw, As, Ap, An, Ae, b, n)
        Ap.fill(0)
        b.fill(0)        

        for i in range(n+Nj):

            walls = boundary(i, Nj, Ni+1)

            if "V" in walls:
                Ap[i] = 1
                Aw[i] = Ae[i] = As[i] = An[i] = 0
                b[i] = uw if "W" in walls else ue
            else:
                Fw = rho * dy * 0.5 * (U[i] + U[i-Nj])
                Fe = rho * dy * 0.5 * (U[i] + U[i+Nj])
                Aw[i] = -Dx - max(0, Fw)
                Ae[i] = -Dx - max(0,-Fe)
                As[i] = An[i] = 0

                if "S" in walls:
                    tmp = (8/3) * Dy + max(0, rho*dx*vs)
                    Ap[i] += tmp
                    b[i] += tmp * us
                    An[i] -= Dy/3
                else:
                    Fs = rho * dx * 0.5 * (V[i + int(i/Nj)] + V[i + int(i/Nj) - Nj - 1])
                    As[i] -= Dy + max(0, Fs)

                if "N" in walls:
                    tmp = (8/3) * Dy + max(0, -rho*dx*vn)
                    Ap[i] += tmp
                    b[i] += tmp * un
                    As[i] -= Dy/3
                else:
                    Fn = rho * dx * 0.5 * (V[i + int(i/Nj) + 1] + V[i + int(i/Nj) - Nj])
                    An[i] -= Dy + max(0,-Fn)

                Ap[i] -= Aw[i] + As[i] + An[i] + Ae[i]
                b[i]  += dy * (P[i-Nj] - P[i])
            
            U_S[i] = (b[i] - (Aw[i] * pad(U, i-Nj) + As[i] * pad(U, i-1) + An[i] * pad(U, i+1) + Ae[i] * pad(U, i+Nj)))/Ap[i]
            de[i]  = dy/Ap[i]

        #U_S = SIP(Aw, As, Ap, An, Ae, b, n+Nj, 10)
        Ap.fill(0)
        b.fill(0)
        
        for i in range(n+Ni):

            walls = boundary(i, Nj+1, Ni)

            if "H" in walls:
                Ap[i] = 1
                Aw[i] = Ae[i] = As[i] = An[i] = 0
                b[i] = vs if "S" in walls else vn
            else:
                Fs = rho * dx * 0.5 * (V[i] + V[i - 1])
                Fn = rho * dx * 0.5 * (V[i] + V[i + 1])
                As[i] = -Dy - max(0, Fs)
                An[i] = -Dy - max(0,-Fn)
                Aw[i] = Ae[i] = 0

                if "W" in walls:
                    tmp = (8/3) * Dx + max(0, rho*dy*uw)
                    Ap[i] += tmp
                    b[i] += tmp * vw
                    Ae[i] -= Dx/3
                else:
                    Fw = rho * dx * 0.5 * (U[i - int(i/(Nj+1))] + U[i - int(i/(Nj+1)) - 1])
                    Aw[i] -= Dx + max(0, Fw)

                if "E" in walls:
                    tmp = (8/3) * Dx + max(0, rho*dy*ue)
                    Ap[i] += tmp
                    b[i] += tmp * ve
                    Ae[i] -= Dx/3
                else:
                    Fe = rho * dx * 0.5 * (U[i - int(i/(Nj+1)) + Nj] + U[i - int(i/(Nj+1)) + Nj - 1])
                    Ae[i] -= Dx + max(0,-Fe)

                Ap[i] -= Aw[i] + As[i] + An[i] + Ae[i]
                b[i]  += dx * (P[i - int(i/(Nj+1)) - 1] - P[i - int(i/(Nj+1))])
            
            V_S[i] = (b[i] - (Aw[i] * pad(V, i-1-Nj) + As[i] * pad(V, i-1) + An[i] * pad(V, i+1) + Ae[i] * pad(V, i+1+Nj)))/Ap[i]
            dn[i]  = dx/Ap[i]        
        
        #V_S = SIP(Aw, As, Ap, An, Ae, b, n+Ni, 10)

        U = U_S
        V = V_S

        err_src = 0
        for k in range(n):

            walls = boundary(k, Nj, Ni)

            Aw[k] = 0 if "W" in walls else -rho * dy * de[k]
            Ae[k] = 0 if "E" in walls else -rho * dy * de[k+Nj]
            As[k] = 0 if "S" in walls else -rho * dx * dn[k + int(k/Nj)]
            An[k] = 0 if "N" in walls else -rho * dx * dn[k + int(k/Nj) + 1]

            Ap[k] = -(Aw[k] + As[k] + An[k] + Ae[k])
            b[k]  = rho * dy * (U[k] - U[k+Nj]) + rho * dx * (V[k + int(k/Nj)] - V[k + int(k/Nj) + 1])
            err_src += b[k]**2

        p = SIP(Aw, As, Ap, An, Ae, b, n)
        err_src = err_src ** 0.5

        for i in range(n+Nj):

            if Nj <= i < n:
                U[i] += w * de[i] * (p[i-Nj] - p[i])

        for j in range(n+Ni):

            if j % (Nj+1) != 0 and (j+1) % (Nj+1) != 0:
                V[j] += w * dn[j] * (p[j - int(j/(Nj+1)) - 1] - p[j - int(j/(Nj+1))])

        itr += 1
        print(itr, err_src)

    export(U, V, P)


NS_steady()






"""
Bogoliubov (BdG) correction to the lattice Hartree soliton -- the standard leading
beyond-mean-field theory, used here as an *analytic* anchor for the NQS.

For H = sum_xy h_xy b+_x b_y + 1/2 sum_xy W_xy b+_x b+_y b_y b_x, expanding
b_x = sqrt(N) phi_x + db_x with db orthogonal to phi gives

    A = Q [ h + diag(g U) + g W o (phi phi^T) - mu ] Q ,  U_x = sum_y W_xy phi_y^2, g = N-1
    B = Q [ g W o (phi phi^T) ] Q ,                       Q = 1 - |phi><phi|

(kinetic + direct/Hartree + exchange in A; B is the anomalous pair term).  Then

    dE_Bog = 1/2 ( sum_k omega_k - Tr A ) ,   omega^2 = eig[(A+B)(A-B)] ,
    depletion = sum_k |v_k|^2 .
"""
import numpy as np


def bogoliubov(lat, phi, N, use_Nm1=True):
    K = lat.K
    p = np.asarray(phi, dtype=np.float64).reshape(-1)
    p = p / np.linalg.norm(p)
    W = np.asarray(lat.W, dtype=np.float64)
    h = np.asarray(lat.h, dtype=np.float64)
    g = (N - 1.0) if use_Nm1 else float(N)
    U = W @ (p ** 2)
    mu = float(p @ h @ p + g * ((p ** 2) @ U))
    Wpp = W * np.outer(p, p)
    Q = np.eye(K) - np.outer(p, p)
    A = Q @ (h + np.diag(g * U) + g * Wpp - mu * np.eye(K)) @ Q
    B = Q @ (g * Wpp) @ Q
    A = 0.5 * (A + A.T); B = 0.5 * (B + B.T)
    # full symplectic eigenproblem (K is small in the radial model)
    M = np.block([[A, B], [-B, -A]])
    w, V = np.linalg.eig(M)
    sel = np.argsort(-w.real)[: K - 1]          # positive branch, minus the zero mode
    omega = np.sort(w.real[w.real > 1e-10])[::-1]
    dE = 0.5 * (omega.sum() - np.trace(A))
    depl = 0.0
    for i in sel:
        if w[i].real <= 1e-10:
            continue
        u, v = V[:K, i].real, V[K:, i].real
        nrm = u @ u - v @ v
        if nrm > 1e-12:
            depl += (v @ v) / nrm
    return dict(dE=float(dE), omega=omega, mu=mu, depletion=float(depl))

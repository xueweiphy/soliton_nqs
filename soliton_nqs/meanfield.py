"""
Lattice mean-field (Hartree / Gross–Pitaevskii) ground state.

Product state |a>^{(x)N}, i.e. the Fock state with all N bosons in the
orbital a_j:

    E_MF(a) = N a^T h a + 1/2 N (N-1) sum_ij W_ij a_i^2 a_j^2 ,   sum a_j^2 = 1.

This is the discretised Schrödinger–Poisson soliton, on exactly the same
lattice as the quantum problem, so it is the right benchmark for the NQS.
"""
import numpy as np
from scipy.optimize import minimize


def mean_field(lat, a0=None, tol=1e-12):
    N, K, h, W = lat.N, lat.K, lat.h, lat.W
    if a0 is None:
        # smooth initial guess: gaussian in u = r psi, width ~ classical soliton
        a0 = np.sqrt(lat.vol) * np.exp(-0.5 * (lat.r / lat.rhalf_hartree)**2)
    v0 = a0 / np.linalg.norm(a0)

    def energy_grad(v):
        nv = np.linalg.norm(v)
        a = v / nv
        rho = a * a
        Wrho = W @ rho
        E = N * a @ h @ a + 0.5 * N * (N - 1) * rho @ Wrho
        gE_a = 2 * N * h @ a + 2 * N * (N - 1) * Wrho * a
        # chain rule through normalisation
        g = (gE_a - a * (a @ gE_a)) / nv
        return E, g

    res = minimize(energy_grad, v0, jac=True, method='L-BFGS-B',
                   options=dict(maxiter=5000, ftol=tol, gtol=1e-10))
    a = res.x / np.linalg.norm(res.x)
    a *= np.sign(a[np.argmax(np.abs(a))])
    E = res.fun
    # chemical potential mu = dE/dN evaluated with GP operator
    Wrho = W @ (a * a)
    mu = a @ h @ a + (N - 1) * (a * a) @ Wrho
    return dict(E=E, a=a, mu=mu, n_mean=N * a * a, density=N * a * a / lat.vol,
               converged=res.success)

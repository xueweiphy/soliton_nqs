"""
Soliton + gas mode basis for the study of soliton growth / evaporation.

Single-particle orbitals = l=0 eigenmodes of the Schrödinger operator in the
soliton's own gravitational potential inside a spherical box of radius R,

    h_MF = -1/2 Lap + Phi_s(r),        u(0) = u(R) = 0 ,

("Schrödinger modes" of the AxionNormalModes paper).  The lowest one is the
soliton orbital chi itself; the excited bound s-states and the box-discretised
continuum states are the *gas modes*.

The soliton is the standard soliton (chi0list.dat) rescaled so that its
Hartree potential corresponds to N_s quanta:  S = (N_s - 1)/mu_0  (this makes
the Fock state |N_s, 0, ...> the self-consistent Hartree state up to grid
errors, see check_hartree()).

Second-quantised Hamiltonian in this basis (real orbitals):

    H = sum_{kk'} h0_{kk'} b_k^+ b_k'  +  1/2 sum_{ijkl} U[(i,l),(j,k)] b_i^+ b_j^+ b_k b_l

    h0_{kk'} = <k| -1/2 Lap |k'>  =  eps_k delta_kk' - <k|Phi_s|k'>       (bare kinetic)
    U[(a,b),(c,d)] = - int int  phi_a phi_b (x)  phi_c phi_d (x')  / (4 pi |x-x'|)
                   = - int dr int dr'  rho_ab(r) rho_cd(r') / (4 pi max(r,r'))   (all s-waves)

U is stored as a symmetric (K^2 x K^2) "pair" matrix.  Units hbar = m = 4 pi G = 1.
"""
import os
import numpy as np
from scipy.linalg import eigh_tridiagonal

from .lattice import MU0, EPS0, E_STD, RHALF0


def standard_soliton():
    """r, chi0(r), Phi0(r) of the standard soliton (chi0(0)=1) from chi0list.dat,
    with Phi0 obtained by integrating Poisson: Phi0' = M(r)/(4 pi r^2), Phi0(inf)=0."""
    rr, chi = np.loadtxt(os.path.join(os.path.dirname(__file__), 'chi0list.dat')).T
    r = np.concatenate([[0.0], rr])
    chi = np.concatenate([[1.0], chi])
    dM = 4 * np.pi * r**2 * chi**2
    M = np.concatenate([[0.0], np.cumsum(0.5 * (dM[1:] + dM[:-1]) * np.diff(r))])
    g = M / (4 * np.pi * np.maximum(r, 1e-12)**2)               # dPhi/dr
    # Phi(r) = -M_tot/(4 pi R_end) - int_r^{R_end} g dr'
    Phi_end = -M[-1] / (4 * np.pi * r[-1])
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(r))])
    Phi = Phi_end - (integ[-1] - integ)
    return r, chi, Phi, M[-1]


class ModeBasis:
    def __init__(self, N_s, K, R_factor=8.0, n_grid=3000, S=None):
        """N_s: soliton quanta defining the potential (S=(N_s-1)/mu_0 unless S given);
        K: number of modes (soliton + K-1 gas modes); R = R_factor * r_1/2."""
        self.N_s, self.K = N_s, K
        self.S = (N_s - 1) / MU0 if S is None else S
        S = self.S
        self.rhalf = RHALF0 / S
        self.R = R_factor * self.rhalf
        r0, chi0, Phi0, _ = standard_soliton()
        # rescaled soliton: chi(r) = S^2 chi0(S r), Phi(r) = S^2 Phi0(S r)
        r = np.linspace(0.0, self.R, n_grid + 1)[1:]                # r_1..R (exclude 0)
        self.r = r; self.dr = r[1] - r[0]
        Sr = S * r
        self.chi = S**2 * np.interp(Sr, r0, chi0, right=0.0)
        Phi_far = -(S * MU0) / (4 * np.pi * r)                       # -M/(4 pi r) beyond the table
        self.Phi = np.where(Sr <= r0[-1], S**2 * np.interp(Sr, r0, Phi0), Phi_far)
        # --- l=0 Schrödinger modes: -1/2 u'' + Phi u = eps u, u(0)=u(R)=0
        n = len(r) - 1                                                # interior points r_1..r_{n} (u(R)=0 at r_{n+1})
        d = 1.0 / self.dr**2 + self.Phi[:n]
        e = -0.5 / self.dr**2 * np.ones(n - 1)
        eps, U = eigh_tridiagonal(d, e, select='i', select_range=(0, K - 1))
        u = np.vstack([U, np.zeros((1, K))])                          # append u(R)=0
        phi = u / r[:, None]
        norm = np.sqrt(np.sum(4 * np.pi * r[:, None]**2 * phi**2, axis=0) * self.dr)
        phi = phi / norm
        phi *= np.sign(phi[np.argmax(np.abs(phi), axis=0), np.arange(K)])
        self.eps, self.phi = eps, phi                                 # (K,), (n_grid, K)
        self.n_bound = int(np.sum(eps < 0))
        # --- one-body: h0 = eps delta - <k|Phi|k'>
        w = 4 * np.pi * r**2 * self.dr
        PhiM = (phi * (w * self.Phi)[:, None]).T @ phi
        self.h0 = np.diag(eps) - PhiM
        # --- two-body pair matrix
        rho = np.einsum('ra,rb->rab', phi, phi).reshape(len(r), K * K)   # rho_ab(r)
        A = rho * w[:, None]                                          # includes measure
        G = -1.0 / (4 * np.pi * np.maximum.outer(r, r))
        self.U = A.T @ G @ A                                          # (K^2, K^2), index a*K+b
        self.U = 0.5 * (self.U + self.U.T)
        # overlap of soliton orbital with mode 0
        chi_n = self.chi / np.sqrt(np.sum(w * self.chi**2))
        self.overlap_soliton_mode0 = float(np.sum(w * chi_n * phi[:, 0]))

    # ------------------------------------------------------------------
    def U4(self, i, j, k, l):
        """Matrix element in H_int = 1/2 sum U4(i,j,k,l) b_i^+ b_j^+ b_k b_l:
        particle at x: i -> l, particle at x': j -> k."""
        K = self.K
        return self.U[i * K + l, j * K + k]

    def hartree_energy(self, N):
        """Energy of the Fock state |N,0,...,0> (all quanta in mode 0)."""
        return N * self.h0[0, 0] + 0.5 * N * (N - 1) * self.U4(0, 0, 0, 0)

    def E_hartree_continuum(self, N):
        return E_STD * ((N - 1) / MU0)**2 * N / MU0

    def describe(self):
        return (f"ModeBasis: soliton for N_s={self.N_s} (S={self.S:.4f}, r_1/2={self.rhalf:.3g}), "
                f"box R={self.R:.3g}, K={self.K} l=0 modes ({self.n_bound} bound), "
                f"eps = {np.array2string(self.eps, precision=4)}; "
                f"soliton chemical potential eps0*S^2 = {EPS0*self.S**2:.4f}; "
                f"|<chi|mode0>| = {self.overlap_soliton_mode0:.5f}")

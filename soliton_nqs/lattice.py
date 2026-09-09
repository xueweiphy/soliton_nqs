"""
Radial-lattice, second-quantized Schrödinger–Poisson (SP) Hamiltonian.

Units follow the AxionSoliton project:  hbar = m = 4 pi G = 1, so that the
classical (mean-field) limit is

    i d_t psi + 1/2 Lap psi - Phi psi = 0 ,     Lap Phi = |psi|^2 ,

with the "standard soliton"  chi_0(0)=1,  mu_0 = 25.9,  eps_0 = -0.692.

Quantum problem
---------------
N identical bosons with pair potential  V(x,x') = -1/(4 pi |x-x'|)
(this is  -G m^2/|x-x'|  with 4 pi G = m = 1):

    H = sum_a p_a^2/2  -  sum_{a<b} 1/(4 pi |x_a - x_b|) .

Spherical-symmetry ansatz + radial discretisation
-------------------------------------------------
Every particle is restricted to an l=0 orbital.  The radial direction is
divided into K shells (cells) with edges 0 = e_0 < e_1 < ... < e_K = R,
centres r_j and widths D_j = e_j - e_{j-1}.  Each shell is one bosonic
mode b_j.  Writing u = r*psi and a_j = sqrt(4 pi D_j) u(r_j) (so that
sum_j a_j^2 = 1), the finite-volume kinetic energy  T = 2 pi int |u'|^2 dr
becomes the tridiagonal one-body matrix

    h_jj      = 1/2 [ 1/(D_j (r_j - r_{j-1})) + 1/(D_j (r_{j+1} - r_j)) ]
    h_{j,j+1} = -1/2 / ( sqrt(D_j D_{j+1}) (r_{j+1} - r_j) )         (< 0)

with hard walls u(0) = u(R) = 0.  For a uniform grid this is the usual
h_jj = 1/dr^2, h_{j,j+1} = -1/(2 dr^2).

For spherically symmetric shells the monopole part of 1/|x-x'| is
1/max(r,r'), so gravity is a pure density–density term

    W_ij = -1/(4 pi max(r_i, r_j)) .

The lattice Hamiltonian is a 1D Bose–Hubbard-like chain with long-range
attraction:

    H = sum_ij h_ij b_i^+ b_j  +  1/2 sum_ij W_ij ( n_i n_j - delta_ij n_i ) .

All off-diagonal elements are negative, so (Perron–Frobenius) the ground
state has real positive amplitudes in the Fock basis and a positive ansatz
psi(n) = sqrt(p(n)) is exact in principle.

Grid choice
-----------
A coarse *uniform* grid is dangerous: the finite-difference kinetic energy
per particle is bounded by ~2/dr^2, whereas putting all N particles in the
innermost shell gains ~ N/(8 pi dr).  For dr > ~4 pi/N the lattice model
"collapses" onto the first site, which is a discretisation artefact.  A
*logarithmic* grid (D_j ~ c r_j) cures this: on such a grid the collapse
energy is bounded by ~ -N (N-1)^2 c^2 /(256 pi^2), negligible for c <~ 0.3,
while the outer cells are still large enough to cover the soliton with only
K = 8-32 modes.  Default: log grid.

Scaling
-------
The classical scaling symmetry maps the standard soliton (N_0 = 25.9) to a
soliton with N particles via S = N/25.9:  E -> S^3 E_std,  r -> r/S,
where  E_std = eps_0*mu_0/3 = -5.97  is the total classical energy of the
standard soliton (virial theorem: E = eps N/3 for a purely gravitational
soliton).
"""
from dataclasses import dataclass, field
import numpy as np
import os

MU0 = 25.9          # dimensionless mass of the standard soliton
EPS0 = -0.692       # its chemical potential (binding energy per particle)
RHALF0 = 1.30       # half-density radius of the standard soliton (chi_0^2 = 1/2)
E_STD = EPS0 * MU0 / 3.0   # total classical energy of the standard soliton


def make_edges(K, r_max, grid='log', r_min=None):
    """Cell edges e_0=0 < e_1 < ... < e_K = r_max."""
    if grid == 'uniform':
        return np.linspace(0.0, r_max, K + 1)
    if grid == 'log':
        if r_min is None:
            r_min = r_max * (1.0 / K) ** 1.5      # innermost edge
        inner = np.geomspace(r_min, r_max, K)
        return np.concatenate([[0.0], inner])
    raise ValueError(grid)


@dataclass
class RadialLattice:
    N: int                  # number of bosons
    K: int                  # number of radial modes (shells)
    r_max: float = None     # box radius R  (default 6 x Hartree-soliton half-radius; the exponential tail needs it)
    grid: str = 'log'       # 'log' or 'uniform'
    r_min: float = None     # innermost edge for the log grid (default 12/N, safe against collapse)
    edges: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        K = self.K
        if self.r_max is None:
            self.r_max = 6.0 * self.rhalf_hartree
        if self.r_min is None and self.grid == 'log':
            self.r_min = 12.0 / self.N
        if self.edges is None:
            self.edges = make_edges(K, self.r_max, self.grid, self.r_min)
        e = self.edges
        self.R = e[-1]
        self.width = np.diff(e)                       # D_j
        self.r = 0.5 * (e[1:] + e[:-1])               # shell centres r_j
        # nodes including the two boundary points r=0 and r=R
        rn = np.concatenate([[0.0], self.r, [self.R]])
        dn = np.diff(rn)                              # r_{j+1}-r_j, length K+1
        D = self.width
        h = np.zeros((K, K))
        for j in range(K):
            h[j, j] = 0.5 * (1.0 / (D[j] * dn[j]) + 1.0 / (D[j] * dn[j + 1]))
        for j in range(K - 1):
            off = -0.5 / (np.sqrt(D[j] * D[j + 1]) * dn[j + 1])
            h[j, j + 1] = h[j + 1, j] = off
        self.h = h
        self.t = -np.diag(h, 1)                       # positive hopping amplitudes, length K-1
        rmax = np.maximum.outer(self.r, self.r)
        self.W = -1.0 / (4.0 * np.pi * rmax)
        self.vol = 4.0 * np.pi * self.r**2 * D        # cell volumes: density = n_j / vol_j

    # ---- classical reference values for the same N -------------------------
    @property
    def S(self):
        return self.N / MU0

    @property
    def E_classical(self):
        """Continuum mean-field (soliton) energy for N particles."""
        return E_STD * self.S**3

    @property
    def rhalf_classical(self):
        return RHALF0 / self.S

    # ---- finite-N Hartree (product-state) reference ---------------------------
    # For N particles the Hartree orbital obeys the GP equation with coupling
    # (N-1) instead of N, i.e. it is the continuum soliton with S_H = (N-1)/mu_0:
    #     E_H = N [ T + (N-1) V/2 ]  =  (E_std/mu_0) * S_H^2 * N ,   r_1/2 = 1.30 / S_H .
    # This is what the lattice mean-field converges to as K -> infinity, and it
    # is the natural classical benchmark for the quantum ground state.
    @property
    def S_H(self):
        return (self.N - 1) / MU0

    @property
    def E_hartree(self):
        return E_STD * self.S_H**2 * self.N / MU0

    @property
    def rhalf_hartree(self):
        return RHALF0 / self.S_H

    def classical_density(self, r=None, hartree=True, chi0_file=None):
        """Continuum soliton density profile at radii r, from the project's
        chi0list.dat.  hartree=True: finite-N Hartree soliton (S=(N-1)/mu_0,
        normalised to N particles).  hartree=False: naive N-scaled soliton."""
        if r is None:
            r = self.r
        if chi0_file is None:
            chi0_file = os.path.join(os.path.dirname(__file__), 'chi0list.dat')
        rr, chi = np.loadtxt(chi0_file).T
        if hartree:
            S, norm = self.S_H, self.N / (self.N - 1)
        else:
            S, norm = self.S, 1.0
        chi_r = np.interp(S * r, rr, chi, left=chi[0], right=0.0)
        return norm * S**4 * chi_r**2

    # ---- diagonal energy of a Fock state ------------------------------------
    def diag_energy(self, n):
        n = np.asarray(n, dtype=float)
        e1 = n @ np.diag(self.h)
        e2 = 0.5 * np.einsum('...i,ij,...j->...', n, self.W, n) - 0.5 * n @ np.diag(self.W)
        return e1 + e2

    def describe(self):
        return (f"RadialLattice: N={self.N} bosons, K={self.K} radial shells ({self.grid} grid), "
                f"R={self.R:g}, innermost edge {self.edges[1]:.3g}; "
                f"continuum Hartree soliton for this N: E_H={self.E_hartree:.5g}, "
                f"r_1/2={self.rhalf_hartree:.3g}")

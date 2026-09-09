"""
Soliton growth / evaporation dynamics in the soliton+gas mode basis.

* exact many-body evolution (sparse H in the fixed-N Fock basis, expm_multiply)
* classical multimode field (Gross–Pitaevskii in the same truncated basis),
  optionally averaged over random initial phases (the classical analogue of a
  Fock initial state; "truncated-Wigner without vacuum noise")

    H = sum h0_{kk'} b_k^+ b_k' + 1/2 sum U4(i,j,k,l) b_i^+ b_j^+ b_k b_l
    i dc_i/dt = sum_k' h0_{ik'} c_k' + sum_{jkl} U4(i,j,k,l) c_j^* c_k c_l      (classical)
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.integrate import solve_ivp
from math import comb

from .exact import fock_states, _key_weights


class ModeED:
    """Exact many-body treatment of N bosons in the K-mode basis `mb` (ModeBasis)."""

    def __init__(self, mb, N):
        self.mb, self.N, self.K = mb, N, mb.K
        K = self.K
        self.states = fock_states(N, K)                          # (D, K)
        self.D = len(self.states)
        self.wts = _key_weights(N, K)
        self.keys = self.states.astype(np.int64) @ self.wts
        self.order = np.argsort(self.keys); self.sorted = self.keys[self.order]
        self.H = self._build()

    def index(self, st):
        k = np.asarray(st).astype(np.int64) @ self.wts
        pos = np.searchsorted(self.sorted, k)
        return self.order[pos]

    def _annihilate_pair(self, k, l):
        """Sparse matrix of b_k b_l from N-boson states to (N-2)-boson states (as index in
        the N-2 basis)."""
        st = self.states
        amp = np.ones(self.D)
        s2 = st.copy()
        # b_l first
        amp *= np.sqrt(s2[:, l]); s2[:, l] -= 1
        amp *= np.sqrt(np.maximum(s2[:, k], 0)); s2[:, k] -= 1
        ok = amp > 0
        return s2[ok], amp[ok], np.nonzero(ok)[0]

    def _build(self):
        mb, K, D = self.mb, self.K, self.D
        # one-body
        rows, cols, vals = [], [], []
        st = self.states
        for k in range(K):
            for kp in range(K):
                h = mb.h0[k, kp]
                if abs(h) < 1e-14: continue
                if k == kp:
                    rows.append(np.arange(D)); cols.append(np.arange(D)); vals.append(h * st[:, k])
                else:
                    ok = st[:, kp] > 0
                    idx = np.nonzero(ok)[0]
                    s2 = st[idx].copy()
                    amp = np.sqrt(s2[:, kp]); s2[:, kp] -= 1
                    amp = amp * np.sqrt(s2[:, k] + 1); s2[:, k] += 1
                    rows.append(self.index(s2)); cols.append(idx); vals.append(h * amp)
        H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(D, D))
        # two-body: 1/2 sum U4(i,j,k,l) b_i^+ b_j^+ b_k b_l  =  1/2 sum_{(ij),(kl)} U4 (b_j b_i)^+ (b_k b_l)
        # build B_kl : N -> N-2 as sparse (D2 x D)
        if self.N >= 2:
            states2 = fock_states(self.N - 2, K)
            w2 = _key_weights(self.N - 2, K)
            keys2 = states2.astype(np.int64) @ w2
            o2 = np.argsort(keys2); s2sorted = keys2[o2]
            def idx2(s):
                kk = s.astype(np.int64) @ w2
                return o2[np.searchsorted(s2sorted, kk)]
            D2 = len(states2)
            B = {}
            for k in range(K):
                for l in range(K):
                    s2, amp, src = self._annihilate_pair(k, l)
                    B[(k, l)] = sp.csr_matrix((amp, (idx2(s2), src)), shape=(D2, D))
            Hint = sp.csr_matrix((D, D))
            for k in range(K):
                for l in range(K):
                    Bkl = B[(k, l)]
                    C = None
                    for i in range(K):
                        for j in range(K):
                            u = mb.U4(i, j, k, l)
                            if abs(u) < 1e-14: continue
                            term = u * B[(j, i)]
                            C = term if C is None else C + term
                    if C is not None:
                        Hint = Hint + 0.5 * (C.T @ Bkl)
            H = H + Hint
        H = 0.5 * (H + H.T)
        return H.tocsr()

    # ---- states -------------------------------------------------------
    def fock(self, occ):
        v = np.zeros(self.D); v[self.index(np.array(occ))] = 1.0
        return v

    def occupations(self, psi):
        return (np.abs(psi)**2) @ self.states

    def energy(self, psi):
        return np.real(np.vdot(psi, self.H @ psi))

    def evolve(self, psi0, times):
        """Return array (len(times), D) of psi(t) using expm_multiply."""
        times = np.asarray(times)
        out = expm_multiply(-1j * self.H, psi0.astype(complex), start=times[0], stop=times[-1],
                            num=len(times), endpoint=True)
        return out

    def ground_state(self):
        from scipy.sparse.linalg import eigsh
        w, v = eigsh(self.H, k=1, which='SA')
        return w[0], v[:, 0]


# ---------------------------------------------------------------------------
def classical_rhs(mb):
    K = mb.K
    Upair = mb.U.reshape(K, K, K, K)      # U[a*K+b, c*K+d] -> Upair[a,b,c,d]; U4(i,j,k,l)=Upair[i,l,j,k]
    h0 = mb.h0

    def rhs(t, y):
        c = y[:K] + 1j * y[K:]
        # dc_i/dt = -i [ h0 c + sum_{jkl} U4(i,j,k,l) c_j^* c_k c_l ]
        # sum_{jk} U4(i,j,k,l) c_j^* c_k = sum_{jk} Upair[i,l,j,k] c_j^* c_k  =: M[i,l]
        M = np.einsum('iljk,j,k->il', Upair, np.conj(c), c)
        dc = -1j * (h0 @ c + M @ c)
        return np.concatenate([dc.real, dc.imag])
    return rhs


def classical_evolve(mb, c0, times, rtol=1e-9, atol=1e-11):
    """Evolve complex mode amplitudes c(t) (sum |c|^2 = N) with the truncated GP equation."""
    K = mb.K
    y0 = np.concatenate([np.real(c0), np.imag(c0)])
    sol = solve_ivp(classical_rhs(mb), (times[0], times[-1]), y0, t_eval=times, rtol=rtol, atol=atol,
                    method='DOP853')
    c = sol.y[:K] + 1j * sol.y[K:]
    return c.T                                                # (T, K)


def classical_energy(mb, c):
    K = mb.K
    Upair = mb.U.reshape(K, K, K, K)
    e1 = np.real(np.conj(c) @ mb.h0 @ c)
    e2 = 0.5 * np.real(np.einsum('iljk,i,j,k,l->', Upair, np.conj(c), np.conj(c), c, c))
    return e1 + e2


def classical_phase_average(mb, occ0, times, n_samples=32, seed=0, wigner=False):
    """Classical evolution from |c_k| = sqrt(n_k) with random phases; returns mean and std of
    |c_k(t)|^2 over samples (T, K).
    wigner=True: truncated-Wigner-like sampling, c_k = sqrt(n_k) e^{i phi} + xi_k with complex
    Gaussian vacuum noise <|xi|^2> = 1/2, and 1/2 subtracted from |c_k|^2 (symmetric ordering)."""
    rng = np.random.default_rng(seed)
    occ0 = np.asarray(occ0, dtype=float)
    acc = []
    for s in range(n_samples):
        ph = rng.uniform(0, 2 * np.pi, mb.K)
        c0 = np.sqrt(occ0) * np.exp(1j * ph)
        if wigner:
            c0 = c0 + (rng.normal(size=mb.K) + 1j * rng.normal(size=mb.K)) * 0.5
        n = np.abs(classical_evolve(mb, c0, times))**2
        if wigner:
            n = n - 0.5
        acc.append(n)
    acc = np.array(acc)
    return acc.mean(0), acc.std(0) / np.sqrt(n_samples)


# ===========================================================================
# Soliton FORMATION from a gas on the radial lattice (soliton-free basis)
# ===========================================================================
from .exact import build_hamiltonian


class LatticeED:
    """Exact many-body dynamics of N bosons on the radial lattice `lat` (RadialLattice)."""

    def __init__(self, lat):
        self.lat, self.N, self.K = lat, lat.N, lat.K
        self.H, self.states = build_hamiltonian(lat)
        self.D = len(self.states)
        K, N = self.K, self.N
        self.wts = _key_weights(N, K)
        keys = self.states.astype(np.int64) @ self.wts
        self.order = np.argsort(keys); self.sorted = keys[self.order]
        self._hop = None

    def index(self, st):
        k = np.asarray(st).astype(np.int64) @ self.wts
        return self.order[np.searchsorted(self.sorted, k)]

    # ---- one-body operators b_i^+ b_j within the N sector ----------------
    def hop_ops(self):
        if self._hop is not None: return self._hop
        K, D, st = self.K, self.D, self.states
        ops = {}
        for i in range(K):
            for j in range(K):
                if i == j:
                    ops[(i, j)] = sp.diags(st[:, i].astype(float))
                else:
                    ok = st[:, j] > 0
                    idx = np.nonzero(ok)[0]
                    s2 = st[idx].copy()
                    amp = np.sqrt(s2[:, j]); s2[:, j] -= 1
                    amp = amp * np.sqrt(s2[:, i] + 1); s2[:, i] += 1
                    ops[(i, j)] = sp.csr_matrix((amp, (self.index(s2), idx)), shape=(D, D))
        self._hop = ops
        return ops

    def obdm(self, psi):
        ops = self.hop_ops()
        K = self.K
        rho = np.zeros((K, K), dtype=complex)
        for (i, j), op in ops.items():
            rho[i, j] = np.vdot(psi, op @ psi)
        rho = 0.5 * (rho + rho.conj().T)
        w, v = np.linalg.eigh(rho)
        return rho.real, w[::-1], v[:, ::-1]

    def occupations(self, psi):
        return (np.abs(psi)**2) @ self.states

    def energy(self, psi):
        return np.real(np.vdot(psi, self.H @ psi))

    def evolve(self, psi0, times):
        times = np.asarray(times)
        return expm_multiply(-1j * self.H, psi0.astype(complex), start=times[0], stop=times[-1],
                             num=len(times), endpoint=True)

    def ground_state(self):
        from scipy.sparse.linalg import eigsh
        w, v = eigsh(self.H, k=1, which='SA')
        return w[0], v[:, 0]

    # ---- gas initial states ------------------------------------------------
    def orbital_fock_state(self, orbitals, counts):
        """|Psi> = prod_k (b_{phi_k}^+)^{n_k}/sqrt(n_k!) |0>, with b_{phi}^+ = sum_j phi_j b_j^+,
        orbitals: (K, M) orthonormal columns, counts: (M,) with sum = N.  Built by applying
        creation operators sector by sector."""
        K = self.K
        # start in the 0-particle sector
        vec = {(): 1.0}
        cur_states = np.zeros((1, K), dtype=np.int32); cur = np.array([1.0 + 0j])
        for m, nm in enumerate(counts):
            phi = orbitals[:, m]
            for _ in range(int(nm)):
                Nc = int(cur_states[0].sum())
                new_states = fock_states(Nc + 1, K)
                wn = _key_weights(Nc + 1, K)
                kn = new_states.astype(np.int64) @ wn
                on = np.argsort(kn); sn = kn[on]
                new = np.zeros(len(new_states), dtype=complex)
                for j in range(K):
                    if abs(phi[j]) < 1e-15: continue
                    s2 = cur_states.copy(); amp = np.sqrt(s2[:, j] + 1.0); s2[:, j] += 1
                    idx = on[np.searchsorted(sn, s2.astype(np.int64) @ wn)]
                    np.add.at(new, idx, phi[j] * amp * cur)
                cur_states, cur = new_states, new
        # cur is in the N sector with its own ordering (fock_states(N,K) == self.states)
        cur = cur / np.linalg.norm(cur)
        return cur

    def box_orbitals(self):
        """Eigenvectors of the bare lattice kinetic matrix h (free particle in the box)."""
        w, v = np.linalg.eigh(self.lat.h)
        return w, v


# ---- classical lattice field ---------------------------------------------
def lattice_classical_evolve(lat, a0, times, rtol=1e-9, atol=1e-11):
    """i da_j/dt = sum_k h_jk a_k + (sum_k W_jk |a_k|^2) a_j  (lattice GP), sum|a|^2 = N."""
    K = lat.K
    h, W = lat.h, lat.W
    def rhs(t, y):
        a = y[:K] + 1j * y[K:]
        da = -1j * (h @ a + (W @ np.abs(a)**2) * a)
        return np.concatenate([da.real, da.imag])
    y0 = np.concatenate([a0.real, a0.imag])
    sol = solve_ivp(rhs, (times[0], times[-1]), y0, t_eval=times, rtol=rtol, atol=atol, method='DOP853')
    return (sol.y[:K] + 1j * sol.y[K:]).T


def lattice_classical_ensemble(lat, orbitals, counts, times, n_samples=32, seed=0, wigner=False):
    """Classical field ensemble: a = sum_m sqrt(n_m) e^{i phi_m} orbital_m (+ vacuum noise if
    wigner).  Returns time series of the phase-averaged one-body density matrix
    <a_i^* a_j> (T, K, K) (with 1/2 delta_ij subtracted for wigner)."""
    rng = np.random.default_rng(seed)
    K = lat.K
    acc = np.zeros((len(times), K, K), dtype=complex)
    for s in range(n_samples):
        ph = rng.uniform(0, 2 * np.pi, len(counts))
        a0 = orbitals[:, :len(counts)] @ (np.sqrt(np.asarray(counts, float)) * np.exp(1j * ph))
        if wigner:
            a0 = a0 + (rng.normal(size=K) + 1j * rng.normal(size=K)) * 0.5
        a = lattice_classical_evolve(lat, a0, times)
        acc += np.einsum('ti,tj->tij', np.conj(a), a)
    acc /= n_samples
    if wigner:
        acc -= 0.5 * np.eye(K)[None]
    return acc

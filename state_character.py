"""Is the soliton ground state coherent, Fock, thermal, or squeezed?
Exact diagonalisation in the radial model; all quantities in the natural-orbital basis."""
import numpy as np, scipy.sparse as sp, time
from math import comb
from soliton_nqs import RadialLattice, mean_field
from soliton_nqs.exact import fock_states, _key_weights
from soliton_nqs.dynamics import LatticeED

def annihilation(N, K):
    """b_j : N-sector -> (N-1)-sector, list of K sparse matrices."""
    sN, sM = fock_states(N, K), fock_states(N-1, K)
    wM = _key_weights(N-1, K); keys = sM.astype(np.int64) @ wM
    order = np.argsort(keys); sk = keys[order]
    def idx(st):
        k = st.astype(np.int64) @ wM
        return order[np.searchsorted(sk, k)]
    ops = []
    for j in range(K):
        ok = sN[:, j] > 0; i0 = np.nonzero(ok)[0]
        s2 = sN[i0].copy(); amp = np.sqrt(s2[:, j].astype(float)); s2[:, j] -= 1
        ops.append(sp.csr_matrix((amp, (idx(s2), i0)), shape=(len(sM), len(sN))))
    return ops

def analyse(N, K=10):
    lat = RadialLattice(N=N, K=K)
    ed = LatticeED(lat)
    w, v = sp.linalg.eigsh(ed.H, k=1, which='SA')
    psi = v[:, 0]; E = w[0]
    rho, occ, orb = ed.obdm(psi)                 # occ descending, orb columns
    lam = occ[0]; dep_frac = 1 - lam/N; dep_num = N - lam
    # number operator in the condensate orbital
    ops = ed.hop_ops()
    c = orb[:, 0]
    n0 = sum(c[i]*c[j]*ops[(i, j)] for i in range(K) for j in range(K))
    x = n0 @ psi
    m1 = float(psi @ x); m2 = float(x @ x)
    var = m2 - m1**2
    fano = var/m1
    g2 = 1 + (var - m1)/m1**2                    # <n(n-1)>/<n>^2
    # anomalous pair amplitude  <B_0+ B_0+ B_k B_k> in the natural basis
    b = annihilation(N, K)
    B = [sum(orb[j, k]*b[j] for j in range(K)) for k in range(K)]
    b2 = annihilation(N-1, K)
    B2 = [sum(orb[j, k]*b2[j] for j in range(K)) for k in range(K)]
    p0 = B2[0] @ (B[0] @ psi)
    pair = []
    for k in range(1, K):
        pk = B2[k] @ (B[k] @ psi)
        pair.append(float(p0 @ pk))
    pair = np.array(pair)
    nk = occ[1:]                                  # depletion eigenvalues
    ideal = np.sqrt(nk*(1+nk)) * np.sqrt(lam*(lam-1))   # two-mode squeezed vacuum prediction
    return dict(N=N, E=E, dim=ed.D, lam=lam, dep_frac=dep_frac, dep_num=dep_num,
                m1=m1, var=var, fano=fano, g2=g2, nk=nk, pair=pair, ideal=ideal,
                E_mf=mean_field(lat)['E'])

print(f"{'N':>3} {'dim':>8} {'E':>12} {'dE/|E|':>10} {'N dE/|E|':>9} "
      f"{'depl frac':>10} {'depl QUANTA':>12} {'Fano(n0)':>9} {'g2(n0)':>9}")
res = []
for N in [4, 6, 8, 10, 12]:
    t0 = time.time(); r = analyse(N); res.append(r)
    rel = (r['E'] - r['E_mf'])/abs(r['E'])
    print(f"{N:>3} {r['dim']:>8} {r['E']:>12.6f} {rel:>10.3e} {N*rel:>9.4f} "
          f"{r['dep_frac']:>10.3e} {r['dep_num']:>12.5f} {r['fano']:>9.5f} {r['g2']:>9.6f}"
          f"   [{time.time()-t0:.0f}s]")
np.save('results/state_char.npy', np.array(res, dtype=object), allow_pickle=True)

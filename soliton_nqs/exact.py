"""
Exact diagonalisation in the fixed-N Fock basis (small K, N only).

Number of states = C(N+K-1, K-1); e.g. N=K=8 -> 6435,  N=K=10 -> 92378,
N=K=12 -> 1.35e6 (still fine as sparse, but slow to build in python).
"""
import numpy as np
from itertools import combinations
from math import comb
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def fock_states(N, K):
    """All occupation vectors with sum = N over K modes (stars and bars)."""
    states = np.zeros((comb(N + K - 1, K - 1), K), dtype=np.int32)
    for idx, bars in enumerate(combinations(range(N + K - 1), K - 1)):
        prev = -1
        for j, b in enumerate(bars):
            states[idx, j] = b - prev - 1
            prev = b
        states[idx, K - 1] = N + K - 2 - prev
    return states


def _key_weights(N, K):
    """Mixed-radix (base N+1) weights; unique key for every occupation vector."""
    if (N + 1) ** K >= 2 ** 62:
        raise ValueError("state space too large for int64 keys")
    return (N + 1) ** np.arange(K, dtype=np.int64)


def _index_map(states, N):
    K = states.shape[1]
    keys = states.astype(np.int64) @ _key_weights(N, K)
    order = np.argsort(keys)
    return keys, order


def build_hamiltonian(lat):
    N, K = lat.N, lat.K
    states = fock_states(N, K)
    D = len(states)
    keys, order = _index_map(states, N)
    sorted_keys = keys[order]
    wts = _key_weights(N, K)

    def lookup(st):
        k = st.astype(np.int64) @ wts
        pos = np.searchsorted(sorted_keys, k)
        assert np.all(sorted_keys[pos] == k)
        return order[pos]

    rows, cols, vals = [], [], []
    # diagonal
    diag = lat.diag_energy(states)
    rows.append(np.arange(D)); cols.append(np.arange(D)); vals.append(diag)
    # hopping  j -> j+1 and j+1 -> j
    for j in range(K - 1):
        t = lat.t[j]
        for (src, dst) in ((j, j + 1), (j + 1, j)):
            mask = states[:, src] > 0
            idx = np.nonzero(mask)[0]
            st2 = states[idx].copy()
            st2[:, src] -= 1
            st2[:, dst] += 1
            amp = -t * np.sqrt(states[idx, src] * (states[idx, dst] + 1.0))
            rows.append(lookup(st2)); cols.append(idx); vals.append(amp)
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(D, D))
    return H, states


def exact_ground_state(lat, k=1):
    H, states = build_hamiltonian(lat)
    w, v = eigsh(H, k=k, which='SA')
    i = np.argmin(w)
    psi = v[:, i]
    psi *= np.sign(psi[np.argmax(np.abs(psi))])
    n_mean = (psi**2) @ states
    return dict(E=w[i], psi=psi, states=states, n_mean=n_mean, density=n_mean / lat.vol,
                dim=len(states), spectrum=np.sort(w))

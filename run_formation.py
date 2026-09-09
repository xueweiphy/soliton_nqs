#!/usr/bin/env python3
"""
Soliton FORMATION from a gas: exact many-body dynamics on the radial lattice
(soliton-free basis) versus the classical lattice field.

Initial state: N quanta in free-particle box orbitals (eigenvectors of the bare
kinetic matrix), e.g. --gas 4,4,0 = 4 quanta in the lowest box orbital, 4 in the
second (a Fock state, no phase coherence between orbitals).  Observable:
condensate fraction = largest eigenvalue of <b_i^+ b_j>/N and its natural
orbital, compared with the mean-field soliton.

    python run_formation.py --N 8 --K 10 --gas 4,4,0 --tmax 4000
"""
import argparse, json, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from soliton_nqs import RadialLattice, mean_field
from soliton_nqs.dynamics import LatticeED, lattice_classical_ensemble

C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4', gray='#8a8984')


def cond_frac_series(rho_t):
    cf, nat = [], []
    for rho in rho_t:
        w, v = np.linalg.eigh(0.5 * (rho + rho.conj().T))
        cf.append(w[-1].real); nat.append(v[:, -1])
    return np.array(cf), np.array(nat)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--N', type=int, default=8)
    p.add_argument('--K', type=int, default=10)
    p.add_argument('--gas', default='4,4,0', help='quanta per box orbital (lowest first)')
    p.add_argument('--tmax', type=float, default=4000.)
    p.add_argument('--nt', type=int, default=41)
    p.add_argument('--n_cl', type=int, default=24)
    p.add_argument('--out', default='results')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    counts = [int(x) for x in a.gas.split(',')]
    assert sum(counts) == a.N
    tag = f"form_N{a.N}_K{a.K}_gas{a.gas.replace(',', '-')}"

    lat = RadialLattice(N=a.N, K=a.K); print(lat.describe())
    mf = mean_field(lat)
    ed = LatticeED(lat); print(f"[ED ] Hilbert dim {ed.D}")
    E0, g = ed.ground_state(); _, w0, v0 = ed.obdm(g)
    print(f"[ED ] ground state E0={E0:.5f}, condensate fraction {w0[0]/a.N:.4f}   (E_MF={mf['E']:.5f})")
    wb, vb = ed.box_orbitals()
    print(f"[gas] box orbital energies {np.round(wb[:len(counts)], 4).tolist()}, occupations {counts}")
    psi0 = ed.orbital_fock_state(vb, counts)
    Ein = ed.energy(psi0)
    print(f"[gas] initial energy E={Ein:.5f}  (soliton binding scale E_MF={mf['E']:.4f}); "
          f"soliton period 2pi/|mu| = {2*np.pi/abs(mf['mu']):.0f}")
    times = np.linspace(0, a.tmax, a.nt)
    t0 = time.time(); psis = ed.evolve(psi0, times)
    rho_q = np.array([ed.obdm(ps)[0] for ps in psis])
    cf_q, nat_q = cond_frac_series(rho_q); cf_q /= a.N
    occ_q = np.array([ed.occupations(ps) for ps in psis])
    print(f"[ED ] E(t_max)={ed.energy(psis[-1]):.5f}; condensate fraction {cf_q[0]:.3f} -> {cf_q[-1]:.3f} "
          f"(max {cf_q.max():.3f}); overlap of final natural orbital with MF soliton "
          f"{abs(nat_q[-1] @ mf['a']):.3f}   ({time.time()-t0:.0f}s)")
    t0 = time.time()
    rho_c = lattice_classical_ensemble(lat, vb, counts, times, n_samples=a.n_cl)
    cf_c, nat_c = cond_frac_series(rho_c); cf_c /= a.N
    rho_w = lattice_classical_ensemble(lat, vb, counts, times, n_samples=a.n_cl, wigner=True, seed=1)
    cf_w, nat_w = cond_frac_series(rho_w); cf_w /= a.N
    print(f"[CL ] classical condensate fraction {cf_c[0]:.3f} -> {cf_c[-1]:.3f}; truncated Wigner "
          f"{cf_w[0]:.3f} -> {cf_w[-1]:.3f}   ({time.time()-t0:.0f}s)")

    res = dict(N=a.N, K=a.K, gas=counts, times=times.tolist(), E_initial=Ein, E_ground=E0, E_MF=mf['E'],
               cf_quantum=cf_q.tolist(), cf_classical=cf_c.tolist(), cf_wigner=cf_w.tolist(),
               occ_quantum=occ_q.tolist(), natural_orbital_final=nat_q[-1].real.tolist(),
               a_MF=mf['a'].tolist(), r=lat.r.tolist(), vol=lat.vol.tolist())
    json.dump(res, open(os.path.join(a.out, f"{tag}.json"), 'w'))

    # ---- plots ----------------------------------------------------------
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.9))
    tp = times * abs(mf['mu']) / (2 * np.pi)
    ax = axs[0]
    ax.plot(tp, cf_q, color=C['blue'], lw=1.8, label='exact quantum')
    ax.plot(tp, cf_c, color=C['orange'], lw=1.5, ls='--', label='classical field (phase avg.)')
    ax.plot(tp, cf_w, color=C['aqua'], lw=1.5, ls=':', label='truncated Wigner')
    ax.axhline(w0[0] / a.N, color=C['gray'], lw=0.8, label='ground state')
    ax.set_xlabel('time  [soliton periods 2π/|μ|]'); ax.set_ylabel('condensate fraction  λ_max/N')
    ax.set_title(f'gas {counts} in box orbitals, N={a.N}, K={a.K}'); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25)
    ax = axs[1]
    dens_nat = a.N * cf_q[-1] * nat_q[-1].real**2 / lat.vol
    dens_mf = a.N * mf['a']**2 / lat.vol
    dens_ini = (occ_q[0]) / lat.vol
    dens_fin = (occ_q[-1]) / lat.vol
    ax.plot(lat.r, dens_ini, 's:', color=C['gray'], ms=4, label='initial gas density')
    ax.plot(lat.r, dens_fin, 'o-', color=C['blue'], ms=4, label='final total density')
    ax.plot(lat.r, dens_nat, 'D--', color=C['magenta'], ms=4, label='final condensate (natural orbital)')
    ax.plot(lat.r, dens_mf, '-', color=C['orange'], lw=1.5, label=f'mean-field soliton (N={a.N})')
    ax.set_yscale('log'); ax.set_ylim(1e-5, None); ax.set_xlabel('r'); ax.set_ylabel('density')
    ax.set_title('density profiles'); ax.legend(frameon=False, fontsize=7); ax.grid(alpha=0.25)
    ax = axs[2]
    for k in range(a.K):
        ax.plot(tp, occ_q[:, k], color=plt.cm.viridis(k / (a.K - 1)), lw=1.2, label=f'shell {k+1} (r={lat.r[k]:.1f})')
    ax.set_xlabel('time  [soliton periods]'); ax.set_ylabel('⟨n_j⟩ per shell (exact quantum)')
    ax.set_title('shell occupations'); ax.legend(frameon=False, fontsize=6, ncol=2); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"{tag}.png"), dpi=150); plt.close(fig)
    print("saved", os.path.join(a.out, f"{tag}.png"))


if __name__ == '__main__':
    main()

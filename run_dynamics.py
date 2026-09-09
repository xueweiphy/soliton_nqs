#!/usr/bin/env python3
"""
Soliton growth / evaporation: exact quantum dynamics vs classical field in the
soliton + gas mode basis.

    python run_dynamics.py --Ns 6 --K 8 --gas 3:2 --tmax 3000        # 2 gas quanta in mode 3 (cold gas)
    python run_dynamics.py --Ns 6 --K 8 --gas 7:2 --tmax 3000        # 2 gas quanta in mode 7 (hot gas)
    python run_dynamics.py --Ns 6 --K 8 --gas 3:2,5:2 --tmax 3000    # 4 gas quanta

Outputs in --out: dyn_<tag>.json, dyn_<tag>.png
"""
import argparse, json, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from soliton_nqs.modes import ModeBasis
from soliton_nqs.dynamics import ModeED, classical_phase_average, classical_evolve, classical_energy

C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4', gray='#8a8984')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--Ns', type=int, default=6, help='soliton quanta (defines the potential and mode 0 occupation)')
    p.add_argument('--K', type=int, default=8)
    p.add_argument('--R_factor', type=float, default=8.0)
    p.add_argument('--gas', default='3:2', help='comma list mode:count of gas quanta')
    p.add_argument('--tmax', type=float, default=3000.)
    p.add_argument('--nt', type=int, default=151)
    p.add_argument('--n_cl', type=int, default=48, help='classical phase samples')
    p.add_argument('--out', default='results')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    mb = ModeBasis(a.Ns, a.K, R_factor=a.R_factor)
    print(mb.describe())
    occ = np.zeros(a.K, int); occ[0] = a.Ns
    for item in a.gas.split(','):
        m, c = item.split(':'); occ[int(m)] += int(c)
    N = int(occ.sum())
    tag = f"Ns{a.Ns}_K{a.K}_gas{a.gas.replace(':', 'm').replace(',', '_')}"
    print(f"initial Fock state {occ.tolist()}  (N={N}); soliton period 2pi/|eps_s| = {2*np.pi/abs(mb.eps[0]):.1f}")

    t0 = time.time(); ed = ModeED(mb, N)
    print(f"[ED ] Hilbert dim {ed.D}, nnz {ed.H.nnz}, built in {time.time()-t0:.1f}s")
    E0, g = ed.ground_state()
    print(f"[ED ] ground state E0 = {E0:.5f}, occupations {np.round(ed.occupations(g), 3).tolist()}")
    times = np.linspace(0, a.tmax, a.nt)
    psi0 = ed.fock(occ)
    Ein = ed.energy(psi0)
    t0 = time.time(); psis = ed.evolve(psi0, times)
    nq = np.array([ed.occupations(ps) for ps in psis])
    Efin = ed.energy(psis[-1]); norm = np.linalg.norm(psis[-1])
    print(f"[ED ] E(t=0)={Ein:.5f}  E(t_max)={Efin:.5f}  norm={norm:.6f}   ({time.time()-t0:.1f}s)")
    print(f"[ED ] soliton occupation n0: t=0 {nq[0,0]:.3f} -> t_max {nq[-1,0]:.3f}; time-average {nq[:,0].mean():.3f}")

    t0 = time.time()
    ncl, ncl_err = classical_phase_average(mb, occ, times, n_samples=a.n_cl, wigner=False)
    ntw, ntw_err = classical_phase_average(mb, occ, times, n_samples=a.n_cl, wigner=True, seed=1)
    print(f"[CL ] classical (phase-averaged) n0: t_max {ncl[-1,0]:.3f}, average {ncl[:,0].mean():.3f}; "
          f"truncated Wigner: t_max {ntw[-1,0]:.3f}, average {ntw[:,0].mean():.3f}   ({time.time()-t0:.1f}s)")

    res = dict(Ns=a.Ns, K=a.K, R=mb.R, eps=mb.eps.tolist(), occ0=occ.tolist(), N=N, times=times.tolist(),
               n_quantum=nq.tolist(), n_classical=ncl.tolist(), n_classical_err=ncl_err.tolist(),
               n_wigner=ntw.tolist(), n_wigner_err=ntw_err.tolist(), E_initial=Ein, E_ground=E0,
               dim=ed.D)
    json.dump(res, open(os.path.join(a.out, f"dyn_{tag}.json"), 'w'))

    # ---- plot -----------------------------------------------------------
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axs[0]
    tp = times * abs(mb.eps[0]) / (2 * np.pi)                     # in soliton periods
    ax.plot(tp, nq[:, 0], color=C['blue'], lw=1.8, label='exact quantum')
    ax.plot(tp, ncl[:, 0], color=C['orange'], lw=1.5, ls='--', label='classical field (phase avg.)')
    ax.fill_between(tp, ncl[:, 0] - ncl_err[:, 0], ncl[:, 0] + ncl_err[:, 0], color=C['orange'], alpha=0.2, lw=0)
    ax.plot(tp, ntw[:, 0], color=C['aqua'], lw=1.5, ls=':', label='truncated Wigner')
    ax.axhline(a.Ns, color=C['gray'], lw=0.8)
    ax.set_xlabel('time  [soliton periods 2π/|ε_s|]'); ax.set_ylabel('soliton-mode occupation ⟨n_0⟩')
    ax.set_title(f'soliton with {a.Ns} quanta + gas {a.gas} (N={N}, K={a.K})')
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25)
    ax = axs[1]
    for k in range(a.K):
        col = plt.cm.viridis(k / (a.K - 1))
        ax.plot(tp, nq[:, k], color=col, lw=1.3, label=f'mode {k} (ε={mb.eps[k]:+.3f})')
    ax.set_xlabel('time  [soliton periods]'); ax.set_ylabel('⟨n_k⟩ (exact quantum)')
    ax.set_title('all mode occupations'); ax.set_yscale('log'); ax.set_ylim(1e-3, N * 1.5)
    ax.legend(frameon=False, fontsize=7, ncol=2); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"dyn_{tag}.png"), dpi=150); plt.close(fig)
    print("saved", os.path.join(a.out, f"dyn_{tag}.png"))


if __name__ == '__main__':
    main()

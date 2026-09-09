#!/usr/bin/env python3
"""
Demo: transformer neural quantum state for the quantum axion soliton on a
radial lattice.

    python run_demo.py --N 8  --K 8  --steps 300          # small: checked against exact diagonalisation
    python run_demo.py --N 12 --K 12 --steps 400          # ED still possible (92k states)
    python run_demo.py --N 20 --K 16 --steps 600 --no-ed  # NQS vs mean field only

Outputs (in --out): results_N{N}_K{K}.json, energy_N{N}_K{K}.png, density_N{N}_K{K}.png
"""
import argparse, json, os, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from soliton_nqs import RadialLattice, mean_field, exact_ground_state, TransformerNQS, VMC

C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4', gray='#8a8984')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--N', type=int, default=8)
    p.add_argument('--K', type=int, default=8)
    p.add_argument('--r_max', type=float, default=None)
    p.add_argument('--r_min', type=float, default=None)
    p.add_argument('--grid', default='log')
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--pretrain', type=int, default=0, help='sampled supervised warm start (only needed for --head free)')
    p.add_argument('--head', default='poly', choices=['poly', 'free'])
    p.add_argument('--sr', action='store_true', help='use stochastic reconfiguration (minSR) instead of Adam')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--d_model', type=int, default=32)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--no-ed', action='store_true')
    p.add_argument('--n_measure', type=int, default=8192)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tag = f"N{a.N}_K{a.K}"

    lat = RadialLattice(N=a.N, K=a.K, r_max=a.r_max, r_min=a.r_min, grid=a.grid)
    print(lat.describe())
    res = dict(N=a.N, K=a.K, grid=a.grid, edges=lat.edges.tolist(), r=lat.r.tolist(),
               E_hartree_continuum=lat.E_hartree, rhalf_hartree=lat.rhalf_hartree)

    # ---- mean field --------------------------------------------------------
    mf = mean_field(lat)
    print(f"[MF ] lattice Hartree energy      E_MF = {mf['E']:+.6f}   (continuum Hartree {lat.E_hartree:+.6f})")
    res.update(E_MF=mf['E'], mu_MF=mf['mu'], n_MF=mf['n_mean'].tolist(), a_MF=mf['a'].tolist())

    # ---- exact diagonalisation --------------------------------------------
    ed = None
    if not a.no_ed:
        from math import comb
        dim = comb(a.N + a.K - 1, a.K - 1)
        if dim <= 3_000_000:
            t0 = time.time()
            ed = exact_ground_state(lat)
            print(f"[ED ] exact ground state           E_0  = {ed['E']:+.6f}   (dim {ed['dim']}, {time.time()-t0:.1f}s)")
            res.update(E_ED=float(ed['E']), n_ED=ed['n_mean'].tolist(), dim=int(ed['dim']))
        else:
            print(f"[ED ] skipped (Hilbert space dim {dim:.3g} too large)")

    # ---- transformer NQS ---------------------------------------------------
    model = TransformerNQS(a.K, a.N, d_model=a.d_model, n_heads=a.heads, n_layers=a.layers, d_ff=4 * a.d_model,
                           head=a.head)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[NQS] transformer: d={a.d_model}, {a.layers} layers, {a.heads} heads, {n_par} parameters")
    vmc = VMC(lat, model)
    if a.head == 'poly':
        model.init_mean_field(mf['a'])       # start exactly at the lattice Hartree state
        print("[NQS] initialised exactly at the mean-field product state")
    if a.pretrain:
        print("[NQS] supervised warm start on the mean-field product state")
        vmc.pretrain(mf['a'], steps=a.pretrain, batch=a.batch, log_every=a.pretrain // 2)
    m0 = vmc.measure(2048, obdm=False)
    print(f"[NQS] energy at start of VMC        E    = {m0['E']:+.6f} ± {m0['E_err']:.1e}   var(E_loc)={m0['E_var']:.3g}")
    print("[NQS] variational Monte Carlo" + (" (minSR)" if a.sr else " (Adam)"))
    if a.sr:
        hist = vmc.train_sr(steps=a.steps, batch=a.batch, lr=0.05, log_every=max(1, a.steps // 15))
    else:
        hist = vmc.train(steps=a.steps, batch=a.batch, lr=a.lr, lr_min=a.lr / 10, log_every=max(1, a.steps // 15))
    m = vmc.measure(a.n_measure)
    print(f"[NQS] final                         E    = {m['E']:+.6f} ± {m['E_err']:.1e}   var(E_loc)={m['E_var']:.3g}")
    print(f"[NQS] condensate fraction (largest OBDM eigenvalue / N) = {m['condensate_fraction']:.4f}")
    if ed is not None:
        print(f"      relative error vs ED: {(m['E']-ed['E'])/abs(ed['E']):.2e};   MF error vs ED: {(mf['E']-ed['E'])/abs(ed['E']):.2e}")
    res.update(E_NQS=m['E'], E_NQS_err=m['E_err'], E_NQS_var=m['E_var'], n_NQS=m['n_mean'].tolist(),
               n_var_NQS=m['n_var'].tolist(), condensate_fraction=m['condensate_fraction'],
               obdm_eigs=m['obdm_eigs'].tolist(), natural_orbital=m['natural_orbital'].tolist(),
               history=hist.tolist(), n_params=n_par)
    json.dump(res, open(os.path.join(a.out, f"results_{tag}.json"), 'w'), indent=1)
    torch.save(model.state_dict(), os.path.join(a.out, f"model_{tag}.pt"))

    # ---- plots -------------------------------------------------------------
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(hist[:, 0], hist[:, 1], color=C['blue'], lw=1.5, label='transformer NQS (VMC)')
    ax.fill_between(hist[:, 0], hist[:, 1] - hist[:, 2], hist[:, 1] + hist[:, 2], color=C['blue'], alpha=0.2, lw=0)
    ax.axhline(mf['E'], color=C['orange'], lw=1.5, ls='--', label='lattice mean field (Hartree)')
    if ed is not None:
        ax.axhline(ed['E'], color=C['aqua'], lw=1.5, ls=':', label='exact diagonalisation')
    ax.set_xlabel('VMC step'); ax.set_ylabel('energy  E  [ħ = m = 4πG = 1]')
    ax.set_title(f'Quantum soliton ground state, N={a.N} bosons, K={a.K} radial modes')
    lo = min(hist[-len(hist)//2:, 1].min(), mf['E'], ed['E'] if ed else mf['E'])
    hi = max(hist[len(hist)//10:, 1].max(), mf['E'])
    ax.set_ylim(lo - 0.15 * (hi - lo) - 1e-9, hi + 0.5 * (hi - lo) + 1e-9)
    ax.legend(frameon=False); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"energy_{tag}.png"), dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axs[0]
    rr = np.linspace(0, lat.R, 400)
    ax.plot(rr, lat.classical_density(rr), color=C['gray'], lw=1.5, label='continuum Hartree soliton')
    ax.plot(lat.r, mf['density'], 'o--', color=C['orange'], ms=5, lw=1, label='lattice mean field')
    if ed is not None:
        ax.plot(lat.r, ed['density'], 's:', color=C['aqua'], ms=5, lw=1, label='exact')
    ax.plot(lat.r, m['density'], 'D-', color=C['blue'], ms=5, lw=1.5, label='transformer NQS')
    ax.set_xlabel('r'); ax.set_ylabel('density  ⟨n_j⟩ / V_j'); ax.set_yscale('log')
    ax.set_ylim(bottom=max(1e-4 * m['density'].max(), m['density'][m['density'] > 0].min() * 0.5))
    ax.set_title('radial density profile'); ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25)
    ax = axs[1]
    x = np.arange(1, a.K + 1)
    ax.bar(x - 0.2, mf['n_mean'], 0.4, color=C['orange'], label='⟨n_j⟩ mean field')
    ax.bar(x + 0.2, m['n_mean'], 0.4, color=C['blue'], label='⟨n_j⟩ NQS')
    ax.errorbar(x + 0.2, m['n_mean'], yerr=np.sqrt(m['n_var']), fmt='none', ecolor='k', elinewidth=1, capsize=2,
                label='± std of n_j (NQS)')
    if ed is not None:
        ax.plot(x, ed['n_mean'], 'k_', ms=12, mew=1.5, label='exact')
    ax.set_xlabel('radial mode j'); ax.set_ylabel('occupation'); ax.set_title('mode occupations')
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25, axis='y')
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"density_{tag}.png"), dpi=150); plt.close(fig)
    print("saved", os.path.join(a.out, f"results_{tag}.json"))


if __name__ == '__main__':
    main()

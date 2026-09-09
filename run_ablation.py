#!/usr/bin/env python3
"""
Ablation: why a transformer?  Same lattice, same head, same training protocol; only
the context backbone changes:  none (structured head only) | mlp | rnn | transformer,
plus transformer with a free softmax head.  Reference: exact diagonalisation and Hartree.

    python run_ablation.py --N 8 --K 8 --steps 500 --seeds 2
"""
import argparse, json, os, time
import numpy as np, torch
from soliton_nqs import RadialLattice, mean_field, exact_ground_state, TransformerNQS, VMC


def run(lat, mf, backbone, head, steps, batch, seed, d_model=32, n_layers=2, lr=1e-3):
    torch.manual_seed(seed)
    model = TransformerNQS(lat.K, lat.N, d_model=d_model, n_heads=4, n_layers=n_layers, d_ff=4 * d_model,
                           head=head, backbone=backbone)
    npar = sum(p.numel() for p in model.parameters())
    v = VMC(lat, model)
    if head == 'poly':
        model.init_mean_field(mf['a'])
    else:
        v.pretrain(mf['a'], steps=400, batch=batch, log_every=0)
    t0 = time.time()
    hist = v.train(steps=steps, batch=batch, lr=lr, lr_min=lr / 10, log_every=0)
    dt = time.time() - t0
    m = v.measure(8192, obdm=False)
    return dict(E=m['E'], err=m['E_err'], var=m['E_var'], npar=npar, time=dt,
                last50=float(hist[-50:, 1].mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--N', type=int, default=8); p.add_argument('--K', type=int, default=8)
    p.add_argument('--steps', type=int, default=500); p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--seeds', type=int, default=2); p.add_argument('--no-ed', action='store_true')
    p.add_argument('--out', default='results')
    a = p.parse_args()
    lat = RadialLattice(N=a.N, K=a.K); mf = mean_field(lat)
    print(lat.describe()); print(f"Hartree (lattice MF)  E = {mf['E']:+.6f}")
    Eed = None
    if not a.no_ed:
        ed = exact_ground_state(lat); Eed = float(ed['E']); print(f"exact diagonalisation E = {Eed:+.6f}")
    configs = [('none', 'poly'), ('mlp', 'poly'), ('rnn', 'poly'), ('transformer', 'poly'), ('transformer', 'free')]
    res = dict(N=a.N, K=a.K, E_MF=mf['E'], E_ED=Eed, runs={})
    for bb, hd in configs:
        rs = [run(lat, mf, bb, hd, a.steps, a.batch, s) for s in range(a.seeds)]
        Es = np.array([r['E'] for r in rs]); best = rs[int(np.argmin(Es))]
        key = f"{bb}+{hd}"
        res['runs'][key] = dict(runs=rs, E_best=float(Es.min()), E_mean=float(Es.mean()), E_spread=float(Es.max() - Es.min()))
        rel = (Es.min() - Eed) / abs(Eed) if Eed else float('nan')
        print(f"{key:20s} params={best['npar']:6d}  E_best={Es.min():+.6f} ± {best['err']:.1e}  "
              f"var={best['var']:.2e}  seeds spread={Es.max()-Es.min():.1e}  rel.err vs ED={rel:.2e}  {best['time']:.0f}s/run", flush=True)
    json.dump(res, open(os.path.join(a.out, f"ablation_N{a.N}_K{a.K}.json"), 'w'), indent=1)


if __name__ == '__main__':
    main()

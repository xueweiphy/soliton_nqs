#!/usr/bin/env python3
"""t-VMC validation: collapse of N bosons from the lowest box orbital, transformer TDVP vs exact.
   python run_tvmc_check.py T_END DT BATCH D_MODEL LAYERS TAG     e.g.  400 0.25 512 32 2 d32"""
import numpy as np, torch, time, json, sys
from soliton_nqs import RadialLattice, TransformerNQS, VMC
from soliton_nqs.dynamics import LatticeED
from soliton_nqs.tvmc import TVMC
torch.manual_seed(0); torch.set_num_threads(2)
lat = RadialLattice(N=6, K=8)
ed = LatticeED(lat); wb, vb = ed.box_orbitals(); phi = vb[:,0]*np.sign(vb[np.argmax(abs(vb[:,0])),0])
psi0 = ed.orbital_fock_state(vb, [6])
T_END, DT = float(sys.argv[1]), float(sys.argv[2])
times = np.arange(0, T_END+1e-9, 25.0)
ps = ed.evolve(psi0, times)
ex = {"t": times.tolist(), "occ": [ed.occupations(p).tolist() for p in ps], "cf": [ (ed.obdm(p)[1][0]/6) for p in ps], "E": ed.energy(psi0)}
print("exact occ(t):"); [print("  t=%5.0f"%t, np.round(o,3), " cf=%.3f"%c) for t,o,c in zip(times, ex["occ"], ex["cf"])]
model = TransformerNQS(lat.K, lat.N, d_model=int(sys.argv[4]), n_heads=4, n_layers=int(sys.argv[5]), d_ff=4*int(sys.argv[4]), use_phase=True)
model.init_mean_field(phi); v = VMC(lat, model)
print("params", sum(p.numel() for p in model.parameters()))
tv = TVMC(v, batch=int(sys.argv[3]), diag_shift=1e-4)
t0=time.time(); k,E,var,n = tv.rhs(); print("rhs %.2fs"%(time.time()-t0), "E", E)
rec = {"t": [], "occ": [], "cf": [], "E": []}
def observe(t, n):
    m = v.measure(2048, obdm=True)
    rec["t"].append(t); rec["occ"].append(m['n_mean'].tolist()); rec["cf"].append(m['condensate_fraction']); rec["E"].append(m['E'])
    print("  OBS t=%6.1f"%t, np.round(m['n_mean'],3), "cf=%.3f"%m['condensate_fraction'], "E=%.5f"%m['E'], flush=True)
hist = tv.evolve(T_END, DT, observe=observe, obs_every=int(25/DT), log_every=int(25/DT))
json.dump({"exact": ex, "nqs": rec, "hist": hist.tolist()}, open("results/tvmc_N6_K8_check_%s.json" % sys.argv[6],"w"))

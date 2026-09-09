import numpy as np, torch
from soliton_nqs import *
from soliton_nqs.bogoliubov import bogoliubov
print('Radial (spherical) model -- where exact diagonalisation is available')
print(f"{'N':>4}{'K':>4} {'E_Hartree':>12} {'E_exact':>12} {'dE_exact':>11} {'dE_Bog':>11} {'Bog/exact':>10} {'depl_Bog':>9}")
for N,K in [(6,10),(8,8),(8,10),(12,12)]:
    lat=RadialLattice(N,K); mf=mean_field(lat); ed=exact_ground_state(lat)
    b=bogoliubov(lat, mf['a'], N)
    dex=ed['E']-mf['E']
    print(f'{N:4d}{K:4d} {mf["E"]:12.6f} {ed["E"]:12.6f} {dex:11.3e} {b["dE"]:11.3e} {b["dE"]/dex:10.3f} {b["depletion"]:9.2e}',flush=True)

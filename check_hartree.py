import numpy as np, torch
from soliton_nqs import *
from soliton_nqs.bogoliubov import bogoliubov

def bog_proj(lat, phi, N, extra=None):
    """Bogoliubov with an extra direction projected out (the dphi/dN partner of the
    phase zero mode, which the naive Q = 1-|phi><phi| leaves behind)."""
    K=lat.K; p=np.asarray(phi,float).reshape(-1); p=p/np.linalg.norm(p)
    W=np.asarray(lat.W,float); h=np.asarray(lat.h,float); g=N-1.0
    U=W@(p**2); mu=float(p@h@p+g*((p**2)@U)); Wpp=W*np.outer(p,p)
    Q=np.eye(K)-np.outer(p,p)
    if extra is not None:
        c=np.asarray(extra,float).reshape(-1); c=Q@c; c/=np.linalg.norm(c)
        Q=Q-np.outer(c,c)
    A=Q@(h+np.diag(g*U)+g*Wpp-mu*np.eye(K))@Q; B=Q@(g*Wpp)@Q
    A=0.5*(A+A.T); B=0.5*(B+B.T)
    M=np.block([[A,B],[-B,-A]]); w,V=np.linalg.eig(M)
    pos=np.where(w.real>1e-9)[0]; omega=np.sort(w.real[pos])[::-1]
    dE=0.5*(omega.sum()-np.trace(A)); depl=0.
    for i in pos:
        u,v=V[:K,i].real,V[K:,i].real; nrm=u@u-v@v
        if nrm>1e-12: depl+=(v@v)/nrm
    return dE, depl, omega

# depletion columns are all in QUANTA (N - lambda_max for exact, sum |v_k|^2 for BdG)
print(f"{'N':>4}{'K':>4} {'corr_exact':>11} {'Bog':>11} {'Bog/ex':>7} | {'depl_exact':>10} {'depl_Bog':>9} {'depl_Bog(proj)':>14}")
for N,K in [(8,8),(12,12)]:
    lat=RadialLattice(N,K); mf=mean_field(lat); ed=exact_ground_state(lat)
    # exact depletion from the OBDM
    from soliton_nqs.dynamics import LatticeED
    led=LatticeED(lat); E0,g0=led.ground_state(); rho,wv,vv=led.obdm(g0)
    depl_ex=N-wv[0]          # quanta, same units as the BdG sum |v_k|^2
    # dphi/dN by finite difference: re-run the Hartree minimiser with N+2 on the same grid
    from soliton_nqs.meanfield import mean_field as MF
    class Fake: pass
    f=Fake(); f.N=N+2; f.K=lat.K; f.h=lat.h; f.W=lat.W; f.vol=lat.vol; f.r=lat.r
    f.rhalf_hartree=lat.rhalf_hartree
    ap=MF(f)['a']
    dphi=ap-mf['a']
    dE_p,depl_p,om_p=bog_proj(lat,mf['a'],N,extra=dphi)
    b=bogoliubov(lat,mf['a'],N)
    corr=ed['E']-mf['E']
    print(f'{N:4d}{K:4d} {corr:11.3e} {b["dE"]:11.3e} {b["dE"]/corr:7.3f} | {depl_ex:10.4f} {b["depletion"]:9.4f} {depl_p:14.4f}')
    print(f'      omega(full) min={b["omega"].min():.4f}   omega(projected) min={om_p.min():.4f}   dE(proj)={dE_p:.3e}')

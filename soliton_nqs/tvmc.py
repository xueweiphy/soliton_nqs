"""
Time-dependent VMC (TDVP) for the transformer NQS on the radial lattice.

Ansatz: log psi = 1/2 log p(n) + i phi(n), real parameters theta.  From
i d/dt psi = H psi and d/dt log psi = sum_k O_k dtheta_k (O_k = d log psi/d theta_k,
complex), least squares over real dtheta gives

    Re(S) dtheta/dt = Im(F),   S = <O^* O>_c,  F = <O^* E_loc>_c .

With X = [Re O; Im O] (2B x P, centred, /sqrt(B)) and y = [Im eps; -Re eps]/sqrt(B),
eps = E_loc - <E_loc>:   dtheta/dt = X^T (X X^T + lambda)^(-1) y   (minSR form, 2B x 2B solve).
Integrator: Heun (2 evaluations per step).
"""
import time
import numpy as np
import torch
from torch.func import functional_call, vmap, grad


class TVMC:
    def __init__(self, vmc, diag_shift=1e-4, batch=512, eloc_clip=None):
        self.vmc, self.model = vmc, vmc.model
        assert self.model.use_phase, "TVMC needs use_phase=True"
        self.batch, self.diag_shift, self.eloc_clip = batch, diag_shift, eloc_clip
        model = self.model

        class _Wrap(torch.nn.Module):
            def __init__(s, m): super().__init__(); s.m = m
            def forward(s, n): return s.m.log_psi(n)
        self.wrap = _Wrap(model)

        def f_abs(pd, n1):
            la, ph = functional_call(self.wrap, pd, (n1[None, :],)); return la[0]
        def f_ph(pd, n1):
            la, ph = functional_call(self.wrap, pd, (n1[None, :],)); return ph[0]
        self.g_abs = vmap(grad(f_abs), in_dims=(None, 0))
        self.g_ph = vmap(grad(f_ph), in_dims=(None, 0))

    def _params(self):
        return {('m.' + k): v.detach() for k, v in self.model.named_parameters()}

    def _get_flat(self):
        return torch.cat([p.detach().reshape(-1) for p in self.model.parameters()])

    def _set_flat(self, flat):
        with torch.no_grad():
            off = 0
            for p in self.model.parameters():
                m = p.numel(); p.copy_(flat[off:off + m].reshape(p.shape).to(p.dtype)); off += m

    def rhs(self):
        """Return (dtheta/dt as flat tensor, energy, variance, samples)."""
        model, vmc, B = self.model, self.vmc, self.batch
        model.eval()
        n = model.sample(B)
        e = vmc.local_energy(n)                                   # complex (B,)
        E = e.mean()
        model.train()
        pd = self._params()
        ga = self.g_abs(pd, n); gp = self.g_ph(pd, n)
        Oa = torch.cat([ga[k].reshape(B, -1) for k in pd], 1).double()
        Op = torch.cat([gp[k].reshape(B, -1) for k in pd], 1).double()
        Oa = Oa - Oa.mean(0, keepdim=True); Op = Op - Op.mean(0, keepdim=True)
        eps = (e - E)
        if self.eloc_clip:
            er = eps.real; sig = er.abs().median() * 1.4826 * self.eloc_clip + 1e-12
            eps = torch.complex(er.clamp(-sig, sig), eps.imag.clamp(-sig, sig))
        X = torch.cat([Oa, Op], 0) / np.sqrt(B)                   # (2B, P)
        y = torch.cat([eps.imag, -eps.real], 0) / np.sqrt(B)      # (2B,)
        T = X @ X.T
        T.diagonal().add_(self.diag_shift + 1e-12)
        z = torch.linalg.solve(T, y)
        dtheta = X.T @ z
        return dtheta, E.real.item(), e.real.var().item(), n

    def evolve(self, t_end, dt, observe=None, obs_every=10, log_every=50):
        """Heun integration; observe(t, samples) is called every obs_every steps."""
        t, it = 0.0, 0
        hist = []
        t0 = time.time()
        while t < t_end - 1e-12:
            h = min(dt, t_end - t)
            th0 = self._get_flat()
            k1, E1, var1, n1 = self.rhs()
            self._set_flat(th0 + h * k1)
            k2, E2, var2, _ = self.rhs()
            self._set_flat(th0 + 0.5 * h * (k1 + k2))
            t += h; it += 1
            hist.append((t, E1, var1))
            if observe is not None and (it % obs_every == 0 or it == 1):
                observe(t, n1)
            if log_every and (it % log_every == 0 or it == 1):
                print(f"  t={t:8.2f}  E={E1:+.5f}  var={var1:.3g}  |dtheta|={k1.norm().item():.3g}  {time.time()-t0:6.0f}s")
        return np.array(hist)

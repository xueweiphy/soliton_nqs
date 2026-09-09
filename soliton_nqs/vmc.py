"""
Variational Monte Carlo for the transformer NQS on the radial SP lattice.

Local energy for a sampled Fock state n:

    E_loc(n) = E_diag(n) + sum_{<ij>} (-t) sqrt(n_i (n_j+1)) psi(n - e_i + e_j) / psi(n)

(nearest-neighbour hoppings only, 2(K-1) terms).  Gradient of <E>:

    grad E = 2 Re < (E_loc - <E>)^* d log psi / d theta >
"""
import time
import numpy as np
import torch

from .model import TransformerNQS


def hop_neighbours(n, K, t):
    """n: (B, K) long; t: (K-1,) hopping amplitudes. Return neighbour configs
    (B, 2(K-1), K) and matrix elements (B, 2(K-1)) = -t_bond sqrt(n_src (n_dst+1))."""
    B = n.shape[0]
    src = torch.cat([torch.arange(K - 1), torch.arange(1, K)])       # j->j+1, j+1->j
    dst = torch.cat([torch.arange(1, K), torch.arange(K - 1)])
    M = len(src)
    nb = n[:, None, :].repeat(1, M, 1)
    ar = torch.arange(M)
    n_src = nb[:, ar, src]
    n_dst = nb[:, ar, dst]
    nb[:, ar, src] = n_src - 1
    nb[:, ar, dst] = n_dst + 1
    me = torch.sqrt(n_src.clamp(min=0).double() * (n_dst + 1).double())
    me = torch.where(n_src > 0, me, torch.zeros_like(me))
    tb = torch.cat([t, t])                                            # bond amplitude per hop
    me = -tb[None, :] * me
    nb = torch.where((n_src > 0)[..., None], nb, n[:, None, :])     # keep valid config (weight 0)
    return nb, me


class VMC:
    def __init__(self, lat, model: TransformerNQS, device='cpu'):
        self.lat, self.model, self.device = lat, model.to(device), device
        self.hdiag = torch.tensor(np.diag(lat.h), dtype=torch.float64)
        self.W = torch.tensor(lat.W, dtype=torch.float64)
        self.Wdiag = torch.tensor(np.diag(lat.W), dtype=torch.float64)
        self.t = torch.tensor(lat.t, dtype=torch.float64)

    # ---- energy pieces --------------------------------------------------
    def diag_energy(self, n):
        nf = n.double()
        return nf @ self.hdiag + 0.5 * torch.einsum('bi,ij,bj->b', nf, self.W, nf) - 0.5 * nf @ self.Wdiag

    def local_energy(self, n, logpsi_n=None):
        """Return E_loc (complex tensor, (B,)) and log|psi(n)| (with graph if requested)."""
        K = self.lat.K
        B = n.shape[0]
        nb, me = hop_neighbours(n, K, self.t)
        flat = nb.reshape(-1, K)
        # evaluate all unique configs (samples + neighbours) in one pass
        allc = torch.cat([n, flat], 0)
        uniq, inv = torch.unique(allc, dim=0, return_inverse=True)
        with torch.no_grad():
            la, ph = self.model.log_psi(uniq)
        la_all, ph_all = la[inv], ph[inv]
        la_n, ph_n = la_all[:B], ph_all[:B]
        la_nb = la_all[B:].reshape(B, -1)
        ph_nb = ph_all[B:].reshape(B, -1)
        ratio = torch.exp((la_nb - la_n[:, None]).double()) * torch.exp(1j * (ph_nb - ph_n[:, None]).double())
        e_off = (me * ratio).sum(1)
        e_loc = self.diag_energy(n).to(torch.complex128) + e_off
        return e_loc

    # ---- training -------------------------------------------------------
    def pretrain(self, a_mf, steps=200, batch=512, lr=3e-3, log_every=50):
        """Supervised warm start: fit p(n) to the mean-field multinomial
        Multinomial(N; a_j^2). Cheap and makes VMC start near the soliton."""
        probs = torch.tensor(a_mf**2 / np.sum(a_mf**2), dtype=torch.float32)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        for it in range(steps):
            n = torch.multinomial(probs, self.lat.N * batch, replacement=True).reshape(batch, self.lat.N)
            counts = torch.nn.functional.one_hot(n, self.lat.K).sum(1)      # (batch, K)
            loss = -self.model.log_prob(counts).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if log_every and (it % log_every == 0 or it == steps - 1):
                print(f"  pretrain {it:4d}  NLL={loss.item():.4f}")

    def train(self, steps=500, batch=512, lr=1e-3, lr_min=1e-4, log_every=20, clip=1.0,
              eloc_clip=5.0, callback=None):
        """eloc_clip: robustify the gradient by clipping E_loc outliers at
        median +- eloc_clip * MAD-based sigma (energy *estimate* is unclipped)."""
        model = self.model
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr_min)
        hist = []
        t0 = time.time()
        for it in range(steps):
            model.eval()
            n = model.sample(batch)
            e_loc = self.local_energy(n)
            model.train()
            la, ph = model.log_psi(n)
            E = e_loc.mean()
            var = e_loc.real.var()
            # gradient estimator (real ansatz: only log|psi| carries gradient)
            if eloc_clip:
                er = e_loc.real
                med = er.median(); sig = 1.4826 * (er - med).abs().median() + 1e-12
                ec = torch.clamp(er, med - eloc_clip * sig, med + eloc_clip * sig)
                w = (ec - ec.mean()).to(e_loc.dtype)
            else:
                w = (e_loc - E)
            if model.use_phase:
                loss = 2.0 * (w.conj().real * la.double() - w.conj().imag * ph.double()).mean()
            else:
                loss = 2.0 * (w.real * la.double()).mean()
            opt.zero_grad(); loss.backward()
            if clip: torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step(); sched.step()
            err = (var / batch).sqrt().item()
            hist.append((it, E.real.item(), err, var.item()))
            if log_every and (it % log_every == 0 or it == steps - 1):
                print(f"  step {it:4d}  E = {E.real.item():+.6f} ± {err:.5f}   var={var.item():.4g}   "
                      f"lr={sched.get_last_lr()[0]:.2e}   {time.time()-t0:6.1f}s")
            if callback: callback(it, E.real.item(), err)
        return np.array(hist)

    # ---- observables ----------------------------------------------------
    @torch.no_grad()
    def measure(self, n_samples=4096, batch=1024, obdm=True):
        model = self.model
        model.eval()
        K, N = self.lat.K, self.lat.N
        es, ns, rho = [], [], torch.zeros(K, K, dtype=torch.float64)
        done = 0
        while done < n_samples:
            b = min(batch, n_samples - done)
            n = model.sample(b)
            es.append(self.local_energy(n).real)
            ns.append(n)
            if obdm:
                rho += self._obdm_batch(n)
            done += b
        e = torch.cat(es); nn_ = torch.cat(ns).double()
        out = dict(E=e.mean().item(), E_err=(e.var() / len(e)).sqrt().item(), E_var=e.var().item(),
                   n_mean=nn_.mean(0).numpy(), n_var=nn_.var(0).numpy(),
                   density=(nn_.mean(0).numpy() / self.lat.vol), samples=torch.cat(ns).numpy())
        if obdm:
            rho = rho / n_samples
            rho = 0.5 * (rho + rho.T)
            w, v = torch.linalg.eigh(rho)
            out['obdm'] = rho.numpy()
            out['condensate_fraction'] = (w.max() / N).item()
            out['natural_orbital'] = v[:, -1].numpy() * np.sign(v[:, -1].numpy()[np.argmax(np.abs(v[:, -1].numpy()))])
            out['obdm_eigs'] = w.numpy()[::-1]
        return out

    @torch.no_grad()
    def _obdm_batch(self, n):
        """<b_i^+ b_j> estimator: rho_ij = < sqrt(n_j (n_i+1)) psi(n - e_j + e_i)/psi(n) >."""
        K = self.lat.K
        B = n.shape[0]
        la_n, _ = self.model.log_psi(n)
        rho = torch.zeros(K, K, dtype=torch.float64)
        rho += torch.diag(n.double().sum(0))
        for j in range(K):
            m = n[:, j] > 0
            if m.sum() == 0: continue
            nm = n[m]
            base = nm.clone(); base[:, j] -= 1
            cfgs = base[:, None, :].repeat(1, K, 1)
            cfgs[:, torch.arange(K), torch.arange(K)] += 1
            flat = cfgs.reshape(-1, K)
            la, _ = self.model.log_psi(flat)
            la = la.reshape(-1, K)
            me = torch.sqrt(nm[:, j].double()[:, None] * (base.double() + 1.0))
            ratio = torch.exp((la - la_n[m][:, None]).double())
            contrib = (me * ratio)                          # (Bm, K): i index
            contrib[:, j] = 0.0                              # diagonal handled above
            rho[:, j] += contrib.sum(0)
        return rho


# ---------------------------------------------------------------------------
# Stochastic reconfiguration (natural gradient) in the minSR formulation
# (Chen & Heyl, Nat. Phys. 2024): with per-sample centred log-derivatives
# O (B x P) and centred local energies eps (B), the update is
#     d theta = - eta * O^T (O O^T + lambda I)^{-1} eps ,
# which needs only a B x B solve and is exact SR when B < P.
# ---------------------------------------------------------------------------
from torch.func import functional_call, vmap, grad


def _train_sr(self, steps=300, batch=1024, lr=0.05, lr_min=None, diag_shift=1e-3,
              log_every=20, eloc_clip=5.0, callback=None):
    model = self.model
    assert not model.use_phase, "minSR implemented for the real (positive) ansatz"
    names = [n for n, _ in model.named_parameters()]
    params = {n: p.detach() for n, p in model.named_parameters()}
    shapes = [p.shape for p in params.values()]
    numels = [p.numel() for p in params.values()]

    def logpsi_single(pdict, n1):
        la, _ = functional_call(model, pdict, (n1[None, :],), {}, )
        return la[0]

    # model.forward is not defined -> wrap log_psi
    class _Wrap(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, n): return s.m.log_psi(n)
    wrap = _Wrap(model)
    wparams = {('m.' + n): p for n, p in params.items()}

    def f(pd, n1):
        la, _ = functional_call(wrap, pd, (n1[None, :],))
        return la[0]
    per_sample_grad = vmap(grad(f), in_dims=(None, 0))

    hist = []
    t0 = time.time()
    if lr_min is None: lr_min = lr
    for it in range(steps):
        eta = lr_min + 0.5 * (lr - lr_min) * (1 + np.cos(np.pi * it / max(1, steps)))
        model.eval()
        n = model.sample(batch)
        e_loc = self.local_energy(n).real
        E = e_loc.mean(); var = e_loc.var()
        model.train()
        wparams = {('m.' + k): v.detach() for k, v in model.named_parameters()}
        g = per_sample_grad(wparams, n)                     # dict of (B, *shape)
        O = torch.cat([g[k].reshape(batch, -1) for k in wparams], dim=1).double()   # (B, P)
        O = O - O.mean(0, keepdim=True)
        if eloc_clip:
            med = e_loc.median(); sig = 1.4826 * (e_loc - med).abs().median() + 1e-12
            ec = torch.clamp(e_loc, med - eloc_clip * sig, med + eloc_clip * sig)
        else:
            ec = e_loc
        eps = (ec - ec.mean()).double()
        O = O / np.sqrt(batch); eps = eps / np.sqrt(batch)
        T = O @ O.T
        T.diagonal().add_(diag_shift + 1e-12)
        x = torch.linalg.solve(T, eps)
        dtheta = -eta * (O.T @ x)                            # (P,)
        # apply update
        with torch.no_grad():
            off = 0
            for (k, p) in model.named_parameters():
                m = p.numel()
                p.add_(dtheta[off:off + m].reshape(p.shape).to(p.dtype))
                off += m
        err = (var / batch).sqrt().item()
        hist.append((it, E.item(), err, var.item()))
        if log_every and (it % log_every == 0 or it == steps - 1):
            print(f"  SR step {it:4d}  E = {E.item():+.6f} ± {err:.5f}   var={var.item():.4g}   "
                  f"eta={eta:.3f}   {time.time()-t0:6.1f}s")
        if callback: callback(it, E.item(), err)
    return np.array(hist)


VMC.train_sr = _train_sr

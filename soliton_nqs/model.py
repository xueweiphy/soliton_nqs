"""
Autoregressive transformer neural quantum state (NQS) over Fock occupation
strings  n = (n_1, ..., n_K),  sum_j n_j = N.

    psi(n) = sqrt(p(n)) * exp(i phi(n)),      p(n) = prod_j p(n_j | n_<j)

The conditional p(n_j | n_<j) is a categorical distribution over
n_j in {0, ..., N - sum_{i<j} n_i}  (masked softmax), so every sample is
automatically a valid N-particle state and psi is exactly normalised.
The last site is deterministic (it takes the remaining particles).

For the SP lattice Hamiltonian the ground state is positive (all hoppings
have negative sign), so the phase head is off by default.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerNQS(nn.Module):
    def __init__(self, K, N, d_model=48, n_heads=4, n_layers=2, d_ff=128,
                 dropout=0.0, use_phase=False, head='poly', poly_order=3, free_head_scale=0.1,
                 backbone='transformer'):
        """head='free': unconstrained logits over n_j = 0..N (standard categorical).
        head='poly': structured logits
              logit(n) = log C(rem, n) + sum_{m=1..M} c_m(context) (n/N)^m * N  + s * free(n)
        i.e. a binomial (mean-field-like) backbone whose parameters depend on the
        context through the transformer, plus a small free residual.  This gives
        smooth, well-behaved tails, which is what keeps the local-energy variance
        under control (a purely free softmax over-populates high-energy tails)."""
        super().__init__()
        self.K, self.N = K, N
        self.d_model = d_model
        self.use_phase = use_phase
        self.head_type = head
        self.poly_order = poly_order
        self.free_head_scale = free_head_scale
        # tokens: 0..N occupation values, plus a START token = N+1
        self.tok_emb = nn.Embedding(N + 2, d_model)
        self.rem_emb = nn.Embedding(N + 1, d_model)     # particles remaining before this site
        self.pos_emb = nn.Parameter(torch.zeros(K, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.backbone = backbone
        if backbone == 'transformer':
            layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                               batch_first=True, norm_first=True,
                                               activation='gelu')
            self.encoder = nn.TransformerEncoder(layer, n_layers)
        elif backbone == 'rnn':
            self.encoder = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
        elif backbone == 'mlp':
            # autoregressive MLP (NADE-like): input = [masked prefix n_<j / N (K), one-hot j (K), rem/N]
            layers, din = [], 2 * K + 1
            for _ in range(max(1, n_layers)):
                layers += [nn.Linear(din, d_model), nn.GELU()]; din = d_model
            self.encoder = nn.Sequential(*layers)
        elif backbone == 'none':
            self.encoder = None            # structured head only: per-site binomial x polynomial, no context
        else:
            raise ValueError(backbone)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N + 1)
        if head == 'poly':
            self.poly_head = nn.Linear(d_model, poly_order)
            nn.init.zeros_(self.poly_head.weight); nn.init.zeros_(self.poly_head.bias)
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
            vals = torch.arange(N + 1, dtype=torch.float64)
            # log C(rem, n) for all rem, n  (−inf where n > rem)
            R = vals[:, None]; nn_ = vals[None, :]
            logC = torch.lgamma(R + 1) - torch.lgamma(nn_ + 1) - torch.lgamma((R - nn_).clamp(min=0) + 1)
            self.register_buffer('logC', logC.float())                       # (N+1 rem, N+1 n)
            self.site_coeff = nn.Parameter(torch.zeros(K, poly_order))       # context-independent part
            self.register_buffer('npow', torch.stack([(vals / N) ** m for m in range(1, poly_order + 1)], 0).float() * N)  # (M, N+1)
        if use_phase:
            self.phase_head = nn.Linear(d_model, N + 1)                # phase contribution phi_j(n_j | context)
            nn.init.zeros_(self.phase_head.weight); nn.init.zeros_(self.phase_head.bias)
            self.register_buffer('sign_offset', torch.zeros(K))        # fixed pi*s_j*n_j term (orbital signs)
        mask = torch.triu(torch.full((K, K), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', mask)

    # ------------------------------------------------------------------
    def _forward_logits(self, n):
        """n: (B, K) long. Returns masked log-conditionals (B, K, N+1) and
        hidden states (B, K, d)."""
        B, K = n.shape
        start = torch.full((B, 1), self.N + 1, dtype=torch.long, device=n.device)
        inp = torch.cat([start, n[:, :-1]], dim=1)                 # shifted right
        cum = torch.cumsum(n, dim=1)
        remaining = self.N - torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-1]], dim=1)
        x = self.tok_emb(inp) + self.rem_emb(remaining) + self.pos_emb[None, :K]
        if self.backbone == 'transformer':
            hid = self.norm(self.encoder(x, mask=self.causal_mask[:K, :K]))
        elif self.backbone == 'rnn':
            hid = self.norm(self.encoder(x)[0])
        elif self.backbone == 'mlp':
            ar = torch.arange(K, device=n.device)
            prefix = (n[:, None, :].float() / self.N) * (ar[None, :, None] > ar[None, None, :]).float()  # (B,K,K)
            onehot = torch.eye(K, device=n.device)[None].expand(B, K, K)
            feats = torch.cat([prefix, onehot, (remaining.float() / self.N)[..., None]], -1)
            hid = self.norm(self.encoder(feats))
        else:
            hid = torch.zeros(B, K, self.d_model, device=n.device)
        if self.head_type == 'poly':
            c = self.poly_head(hid) + self.site_coeff[None, :K]         # (B, K, M)
            logits = self.logC[remaining] + torch.einsum('bkm,mn->bkn', c, self.npow) \
                     + self.free_head_scale * self.head(hid)
        else:
            logits = self.head(hid)                                     # (B, K, N+1)
        # mask n_j > remaining
        vals = torch.arange(self.N + 1, device=n.device)
        invalid = vals[None, None, :] > remaining[:, :, None]
        # last site: forced to take exactly the remainder
        last_force = torch.zeros_like(invalid)
        last_force[:, -1, :] = vals[None, :] != remaining[:, -1:, ]
        invalid = invalid | last_force
        logits = logits.masked_fill(invalid, float('-inf'))
        logp = F.log_softmax(logits, dim=-1)
        return logp, hid

    @torch.no_grad()
    def init_mean_field(self, a):
        """Initialise the state *exactly* at the mean-field product state |a>^N:
        the multinomial Multinomial(N; a_j^2) factorises autoregressively into
        Binomial(rem_j, q_j) with q_j = a_j^2 / sum_{i>=j} a_i^2, i.e. logit
        theta_j = log(a_j^2 / sum_{i>j} a_i^2).  Requires head='poly'."""
        assert self.head_type == 'poly'
        p = torch.as_tensor(a, dtype=torch.float64) ** 2
        p = p / p.sum()
        tail = torch.flip(torch.cumsum(torch.flip(p, [0]), 0), [0])      # sum_{i>=j} p_i
        rest = (tail - p).clamp(min=1e-300)
        theta = torch.log(p.clamp(min=1e-300) / rest)
        theta[-1] = 0.0
        self.site_coeff.zero_()
        self.site_coeff[:, 0] = theta.float().clamp(-60, 60)
        nn.init.zeros_(self.poly_head.weight); nn.init.zeros_(self.poly_head.bias)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        if self.use_phase:
            # product state |a>^N has amplitude sign prod_j sign(a_j)^{n_j}
            self.sign_offset.copy_(torch.tensor([math.pi if x < 0 else 0.0 for x in np.asarray(a)]))
            nn.init.zeros_(self.phase_head.weight); nn.init.zeros_(self.phase_head.bias)

    def log_prob(self, n):
        """log p(n), shape (B,)."""
        logp, _ = self._forward_logits(n)
        return logp.gather(-1, n[..., None]).squeeze(-1).sum(-1)

    def log_psi(self, n):
        """Complex log-amplitude log psi(n) = 0.5 log p + i phi. Returns (logabs, phase)."""
        logp, hid = self._forward_logits(n)
        logabs = 0.5 * logp.gather(-1, n[..., None]).squeeze(-1).sum(-1)
        if self.use_phase:
            ph_all = self.phase_head(hid)                                # (B, K, N+1)
            phase = ph_all.gather(-1, n[..., None]).squeeze(-1).sum(-1) + (n.float() * self.sign_offset[None, :]).sum(-1)
        else:
            phase = torch.zeros_like(logabs)
        return logabs, phase

    @torch.no_grad()
    def sample(self, B):
        """Autoregressive sampling of B configurations, returns (B, K) long."""
        device = self.pos_emb.device
        n = torch.zeros(B, self.K, dtype=torch.long, device=device)
        for j in range(self.K):
            logp, _ = self._forward_logits(n)
            probs = logp[:, j].exp()
            n[:, j] = torch.multinomial(probs, 1).squeeze(-1)
        return n

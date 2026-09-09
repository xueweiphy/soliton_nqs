# Code guide — how `soliton_nqs` is built and how it works

This is a walkthrough of the code, written for someone who wants to understand or extend it
(including the author, six months from now). The README says *what* the project does and
quotes results; this file says *how*, file by file and equation by equation, and lists what is
solid, what is a prototype, and what a reader should check before quoting a number.

Total: ~1900 lines of Python, no framework beyond PyTorch + NumPy/SciPy. Everything runs on a
laptop CPU in minutes for K ≤ 16.

---

## 1. The problem in one paragraph

N identical bosons attract each other gravitationally (pair potential −1/(4π|x−x′|) in units
ħ = m = 4πG = 1). Classically, the mean-field (Hartree / Gross–Pitaevskii / Schrödinger–Poisson)
ground state is the well-known soliton. The question is: **what is the true many-body quantum
ground state at finite N, and how far is it from the mean-field soliton?** The code answers this
on a radial lattice by (i) computing the lattice mean-field state, (ii) computing the exact
ground state by diagonalisation when the Hilbert space is small enough, and (iii) training a
neural-network variational wavefunction (a neural quantum state, NQS) that can go where exact
diagonalisation cannot. The NQS is checked against (ii) and against Bogoliubov theory; the
beyond-mean-field energy it finds is the O(1/N) correlation energy. A second module studies
soliton growth, evaporation and formation as exact quantum dynamics vs. classical field theory.

---

## 2. The main pipeline (`run_demo.py`), step by step

```
RadialLattice(N, K)          lattice.py     grid, one-body h_ij, two-body W_ij, classical refs
      │
      ├─► mean_field(lat)     meanfield.py   Hartree orbital a_j and energy E_MF   (L-BFGS)
      │
      ├─► exact_ground_state  exact.py       sparse H in the fixed-N Fock basis → E_0  (if dim ≲ 3e6)
      │
      └─► TransformerNQS      model.py       autoregressive ψ(n) = √p(n)
            .init_mean_field(a)              start *exactly* at the Hartree product state
            VMC(lat, model)   vmc.py
              .train(...)                    sample → local energy → gradient → Adam
              .measure(...)                  E ± err, ⟨n_j⟩, one-body density matrix
      │
      └─► results/results_N{N}_K{K}.json, energy_*.png, density_*.png
```

Read the files in that order. Each is independent enough to understand on its own.

---

## 3. File map

| File | Lines | Role | Key objects |
|---|---|---|---|
| `soliton_nqs/lattice.py` | 184 | Physics → lattice Hamiltonian | `RadialLattice`, `make_edges`, constants `MU0, EPS0, E_STD, RHALF0` |
| `soliton_nqs/meanfield.py` | 43 | Lattice Hartree (GP) minimiser | `mean_field(lat)` |
| `soliton_nqs/exact.py` | 82 | Exact diagonalisation | `fock_states`, `build_hamiltonian`, `exact_ground_state` |
| `soliton_nqs/model.py` | 169 | The variational ansatz | `TransformerNQS` (+ `backbone='mlp'|'rnn'|'none'`) |
| `soliton_nqs/vmc.py` | 259 | Training and measurement | `VMC.train`, `VMC.train_sr` (minSR), `VMC.measure`, `hop_neighbours` |
| `soliton_nqs/bogoliubov.py` | 47 | Analytic beyond-MF anchor | `bogoliubov(lat, phi, N)` |
| `soliton_nqs/modes.py` | 113 | Soliton + gas single-particle basis | `ModeBasis` (Schrödinger modes, `h0`, pair tensor `U`) |
| `soliton_nqs/dynamics.py` | 315 | Exact and classical time evolution | `ModeED`, `LatticeED`, `classical_evolve`, `classical_phase_average`, `lattice_classical_ensemble` |
| `soliton_nqs/tvmc.py` | 94 | Time-dependent VMC (prototype) | `TVMC.rhs`, `TVMC.evolve` (Heun) |
| `soliton_nqs/chi0list.dat` | — | Tabulated standard soliton profile χ₀(r) | used by `classical_density`, `standard_soliton` |
| `run_demo.py` | 145 | Ground state: MF vs ED vs NQS, plots | main entry point |
| `run_ablation.py` | 58 | Backbone ablation none/MLP/GRU/transformer, poly vs free head | produces `ablation_*.json` |
| `run_dynamics.py` | 96 | Growth/evaporation in the mode basis | `dyn_*.json/png` |
| `run_formation.py` | 115 | Soliton formation from a gas on the lattice | `form_*.json/png` |
| `run_tvmc_check.py` | 28 | t-VMC vs exact evolution | `tvmc_*.json` |
| `check_bogoliubov.py` | 10 | Table: Bogoliubov vs exact correlation energy | prints |
| `check_hartree.py` | ~40 | Zero-mode analysis (projected Bogoliubov, depletion) | prints |
| `state_character.py` | 66 | Is the ground state coherent / Fock / squeezed? (ED) | prints |

---

## 4. `lattice.py` — from the 3D Hamiltonian to a 1D chain

**Approximation.** Every particle is put in an ℓ = 0 orbital, so the problem is radial. The
radial line [0, R] is cut into K shells with edges e₀ = 0 < e₁ < … < e_K = R; each shell is one
bosonic mode `b_j`. Writing u = rψ and a_j = √(4πD_j) u(r_j) with D_j the shell width, the
kinetic energy becomes a **tridiagonal one-body matrix** `lat.h` (nearest-neighbour hopping with
hard walls u(0) = u(R) = 0), and the monopole part of 1/|x−x′| for spherical shells is
1/max(r, r′), so gravity is a **pure density–density term** `lat.W[i,j] = −1/(4π max(r_i, r_j))`.
The result is a Bose–Hubbard-like chain with long-range attraction:

    H = Σ_ij h_ij b_i† b_j + ½ Σ_ij W_ij (n_i n_j − δ_ij n_i)

`lat.t = −diag(h, 1)` stores the positive hopping amplitudes; `lat.vol` the shell volumes
(density = n_j / vol_j); `lat.diag_energy(n)` evaluates the diagonal part for a batch of Fock
states.

**Grid.** `make_edges` gives `'uniform'` or `'log'` (default). The module docstring explains
why uniform is dangerous (the lattice can collapse onto the innermost site — a discretisation
artefact); the log grid with `r_min = 12/N` and `r_max = 6 r₁/₂` avoids it.

**Classical references.** Properties `E_hartree`, `rhalf_hartree` give the *finite-N Hartree*
energy and radius: a product state of N particles obeys the GP equation with coupling (N−1),
i.e. the standard soliton rescaled by S_H = (N−1)/μ₀. This — not the naive N-scaled soliton —
is what the lattice mean field converges to as K → ∞, and it is the correct classical
benchmark. `classical_density(r)` interpolates `chi0list.dat` to produce the continuum profile.

**Sign structure.** All off-diagonal elements of H are negative, so by Perron–Frobenius the
ground state has positive amplitudes in the Fock basis. That is why the ansatz can be
`ψ(n) = √p(n)` with no phase, and why there is no sign problem.

---

## 5. `meanfield.py` — the lattice Hartree state

The product state |a⟩^⊗N (all N bosons in orbital a_j, Σ a_j² = 1) has energy

    E_MF(a) = N aᵀ h a + ½ N(N−1) Σ_ij W_ij a_i² a_j².

`mean_field` minimises this with L-BFGS-B over an unnormalised vector (normalising inside the
objective and pushing the chain rule through). It returns `E`, `a`, the chemical potential
`mu`, `n_mean = N a²` and `density`. It is the discretised Schrödinger–Poisson soliton **on
exactly the lattice the quantum problem uses**, so E_MF vs E_NQS is an apples-to-apples
comparison with no continuum extrapolation.

---

## 6. `exact.py` — exact diagonalisation

`fock_states(N, K)` enumerates all occupation vectors with Σ n_j = N (stars and bars;
dim = C(N+K−1, K−1): 6435 at N=K=8, 92 378 at N=K=12). Each state is mapped to a unique
integer key in base N+1 so that neighbours can be looked up by `searchsorted`.
`build_hamiltonian` assembles the sparse matrix: the diagonal from `lat.diag_energy`, and for
each bond j↔j+1 the hopping elements −t √(n_src (n_dst+1)). `exact_ground_state` calls
`eigsh(..., which='SA')`. This is the ground truth for N=8, K=8 and N=12, K=12.

---

## 7. `model.py` — the ansatz

`TransformerNQS` is an **autoregressive model over occupation strings** n = (n₁, …, n_K):

    p(n) = Π_j p(n_j | n_<j),      ψ(n) = √p(n) · e^{iφ(n)}   (φ ≡ 0 by default)

**Tokens.** Site j sees the previous occupation n_{j−1} (`tok_emb`, with a START token for
j = 1), the number of particles still to place `remaining_j = N − Σ_{i<j} n_i` (`rem_emb`),
and a learned positional embedding. The three are summed into a d_model vector.

**Backbone** (`backbone=` argument, used by the ablation):
- `'transformer'`: `nn.TransformerEncoder` with a causal mask, pre-norm, GELU (default).
- `'rnn'`: a GRU.
- `'mlp'`: a NADE-style MLP whose input is the masked prefix n_<j / N, a one-hot of j, and
  remaining/N.
- `'none'`: no context at all — the head sees only site index and remaining count.

**Head.** The logits over n_j ∈ {0, …, remaining_j} come in two flavours:
- `head='free'`: an unconstrained linear layer → softmax. Standard, and (per the ablation) bad
  here: it over-populates high-energy tails and the local-energy variance is ~100× larger.
- `head='poly'` (default): **structured logits**

      logit(n) = log C(rem, n) + Σ_{m=1..M} c_m(context) · (n/N)^m · N + s · free(n)

  The first term is a binomial; with only the m = 1 coefficient this is *exactly* the
  autoregressive factorisation of a multinomial, i.e. of a mean-field product state. The
  transformer supplies the coefficients `c_m` (via `poly_head`) plus a small free residual
  (`free_head_scale = 0.1`). A per-site context-independent part `site_coeff` is added.

**Masking.** Entries with n > remaining are set to −∞; the last site is forced to take exactly
the remainder. Hence every sample is a valid N-particle state and ψ is exactly normalised —
no Metropolis chain, no autocorrelation, and `sample(B)` is K sequential forward passes.

**`init_mean_field(a)`** is the trick that makes training easy. A multinomial
Multinomial(N; a_j²) factorises autoregressively as Binomial(rem_j, q_j) with
q_j = a_j² / Σ_{i≥j} a_i², so setting `site_coeff[:, 0] = logit(q_j)` and zeroing everything
else puts the network **exactly** at the lattice Hartree state. VMC then only has to learn the
correlation correction, which is ~1 % of the energy.

`log_prob(n)`, `log_psi(n) → (log|ψ|, phase)` and `sample(B)` are the public API used by
`vmc.py`. With `use_phase=True` a second linear head adds a phase per site (needed only for
dynamics, §11).

---

## 8. `vmc.py` — training loop

**Local energy.** For a sampled Fock state n,

    E_loc(n) = E_diag(n) + Σ_{bonds} (−t) √(n_i (n_j+1)) · ψ(n − e_i + e_j) / ψ(n)

Only nearest-neighbour hops exist (2(K−1) neighbours). `hop_neighbours` builds the neighbour
configurations and matrix elements for a whole batch; `local_energy` concatenates samples and
neighbours, evaluates the network **once on the unique configurations** (`torch.unique`), and
scatters back. E_loc is complex in general (real when `use_phase=False`).

**Gradient.** For a normalised autoregressive ansatz the energy gradient is

    ∇E = 2 ⟨ (E_loc − ⟨E⟩) ∇ log|ψ| ⟩

implemented as the surrogate loss `2 · mean(w · log|ψ|)` with `w = E_loc − Ē` detached.
Robustification: E_loc outliers are clipped at median ± 5 MAD-σ *for the gradient only* (the
reported energy is unclipped); gradient-norm clipping at 1.0; Adam with cosine annealing from
`lr` to `lr_min`. `train` returns a history array `(step, E, err, var)`.

**`train_sr`** (activated with `--sr`) is stochastic reconfiguration / natural gradient in the
**minSR** form (Chen & Heyl 2024): with centred per-sample log-derivatives O (B×P) and centred
local energies ε, the update is δθ = −η Oᵀ (O Oᵀ + λ)⁻¹ ε, a B×B solve instead of P×P.
Per-sample gradients come from `torch.func.vmap(grad(...))`. Real ansatz only.

**`measure`** draws `n_samples` fresh samples and returns E ± statistical error, the variance
of E_loc (a quality measure: zero for an eigenstate), ⟨n_j⟩ and its variance, the density, and
optionally the **one-body density matrix** ρ_ij = ⟨b_i† b_j⟩ estimated as
⟨ √(n_j (n_i+1)) ψ(n − e_j + e_i)/ψ(n) ⟩. Its largest eigenvalue / N is the **condensate
fraction**; the corresponding eigenvector is the natural orbital (compare to the Hartree
orbital).

**`pretrain`** fits p(n) to the mean-field multinomial by supervised NLL. Only needed for
`head='free'`, since `init_mean_field` does the same thing exactly for `head='poly'`.

---

## 9. `bogoliubov.py` — the analytic anchor

Standard Bogoliubov–de Gennes expansion around the lattice Hartree orbital φ with coupling
g = N−1: build A (kinetic + Hartree + exchange − μ) and B (anomalous pair term), both projected
with Q = 1 − |φ⟩⟨φ|; solve the symplectic eigenproblem of [[A, B], [−B, −A]]; then

    ΔE_Bog = ½ (Σ_k ω_k − Tr A),      depletion = Σ_k |v_k|².

This gives an independent, non-numerical-optimisation estimate of the correlation energy.
`check_bogoliubov.py` tabulates ΔE_Bog / ΔE_exact = 0.87, 0.90, 0.93 for N = 6, 8, 12 —
approaching 1 from below with N as a leading-order 1/N theory should.

`check_hartree.py` goes one step further: the softest Bogoliubov mode is the partner of the
U(1) phase zero mode (the dφ/dN direction), and projecting it out changes the naive depletion
by a factor ~8. The NQS works at fixed N and never meets this ambiguity. `HARTREE_CHECK.md`
records the numbers.

---

## 10. `run_ablation.py` — is the transformer needed?

Same lattice, same head, same `init_mean_field`, same training protocol; only the backbone
changes. Two seeds each. Result at N = 12, K = 12 (`results/ablation_N12_K12.json`):

| backbone + head | params | mean E over seeds | s/run |
|---|---|---|---|
| none + poly | 1 876 | −0.50248 (stuck at MF) | 66 |
| **mlp + poly** | **3 764** | **−0.50655** | **87** |
| rnn + poly | 14 548 | −0.50672 | 176 |
| transformer + poly | 27 284 | −0.50593 | 290 |
| transformer + free | 27 149 | −0.4972 (var 0.128) | 285 |

Exact: −0.50642. Mean field: −0.50272. Reading: the structured head plus exact Hartree
initialisation is what matters; any context network recovers the correlation energy; the
transformer is not better than an MLP with 7× fewer parameters at this K. Say this plainly
when presenting the project — it is the most transferable finding in the repo.

---

## 11. Dynamics module (`modes.py`, `dynamics.py`, `tvmc.py`)

This is a separate line of work sharing the lattice code. Two bases are used.

**Soliton + gas mode basis (`modes.py`, `ModeED`, `run_dynamics.py`).** Single-particle
orbitals are the ℓ = 0 eigenmodes of −½∇² + Φ_s(r) in a box, where Φ_s is the gravitational
potential of a soliton with N_s quanta (rescaled from `chi0list.dat`). Mode 0 ≈ the soliton
orbital; the rest are bound excited states and box-discretised continuum "gas" modes.
`ModeBasis` builds the bare one-body matrix `h0 = ε δ − ⟨k|Φ_s|k′⟩` and the full four-index
gravity tensor `U` (stored as a K²×K² pair matrix; `U4(i,j,k,l)` indexes it). `ModeED` builds
the sparse many-body H including the two-body term via annihilation-pair operators N → N−2,
and evolves with `expm_multiply`. `classical_evolve` integrates the truncated GP equation
i ċ_i = Σ h0 c + Σ U4 c* c c in the same basis; `classical_phase_average` averages over random
initial phases (classical analogue of a Fock state), with an optional truncated-Wigner vacuum
noise. `run_dynamics.py` puts N_s quanta in mode 0 plus a few gas quanta and compares quantum
vs classical occupation of the soliton mode. Finding in the README: exact quantum evolution
shows net evaporation; the phase-averaged classical field only oscillates; truncated Wigner
tracks the quantum mean.

**Radial lattice basis (`LatticeED`, `run_formation.py`).** Same Hamiltonian as the ground
state problem, but evolved in time from a soliton-free initial state: N quanta in free-particle
box orbitals (eigenvectors of `lat.h`), built as a Fock state by `orbital_fock_state`.
Observable: condensate fraction and natural orbital of the time-dependent one-body density
matrix, compared with the mean-field soliton (overlap 0.97–0.99 within ~1 period).
`lattice_classical_ensemble` is the classical-field comparison.

**t-VMC (`tvmc.py`, `run_tvmc_check.py`) — prototype.** TDVP for the complex ansatz
(`use_phase=True`): Re(S) θ̇ = Im(F), solved in minSR form with a 2B×2B system, Heun
integrator. Validated against `LatticeED` for the collapse of N = 6 quanta from the lowest box
orbital. It tracks the exact dynamics for ~0.6 soliton periods and then loses the growing
inner-shell components. Treat as a working prototype, not a result.

---

## 12. Result files (`results/`)

`results_N{N}_K{K}.json` keys: lattice (`edges`, `r`, `grid`), `E_hartree_continuum`,
`E_MF`, `mu_MF`, `n_MF`, `a_MF`, `E_ED`, `n_ED`, `dim` (if ED ran), `E_NQS`, `E_NQS_err`,
`E_NQS_var`, `n_NQS`, `n_var_NQS`, `condensate_fraction`, `obdm_eigs`, `natural_orbital`,
`history` (list of `[step, E, err, var]`), `n_params`.

The three demo runs shipped:

| file | N | K | E_MF | E_ED | E_NQS | note |
|---|---|---|---|---|---|---|
| `results_N8_K8.json` | 8 | 8 | −0.138920 | −0.140565 | −0.140570 ± 7e-5 | 100 % of correlation energy |
| `results_N12_K12.json` | 12 | 12 | −0.502720 | −0.506423 | −0.506336 ± 3.4e-4 | 98 % |
| `results_N20_K16.json` | 20 | 16 | −2.481885 | — (3×10⁹ states) | −2.4912 ± 0.0019 | 0.37 % below Hartree; condensate fraction 0.9989 |

`ablation_*.json`: see §10. `dyn_*.json`, `form_*.json`, `tvmc_*.json`: time series from the
dynamics scripts; the PNGs beside them are the plots.

---

## 13. Script status — what runs, what is scratch

| Script | Status |
|---|---|
| `run_demo.py`, `run_ablation.py`, `run_dynamics.py`, `run_formation.py` | Production; reproduce everything in `results/` |
| `run_tvmc_check.py` | Prototype; positional CLI args (`T_END DT batch d_model layers tag`) |
| `check_bogoliubov.py`, `check_hartree.py`, `state_character.py` | Analysis scripts that print tables; run from the repo root |
| `check_two_body.py` | **Depends on a separate 3D-lattice code (`nqs3d`) not in this repo**; kept out of the package |

---

## 14. What to check before quoting a number

- **ℓ = 0 restriction.** Every particle is in an s-orbital. Pairs in (ℓ, ℓ) coupled to total
  L = 0 are excluded. The "exact" results are exact *for this model*, not for the full 3D
  problem.
- **Discretisation.** K and the grid control the error; the lattice Hartree energy is within
  3 % (K=8), 0.7 % (K=12), 0.1 % (K=16) of the continuum finite-N Hartree energy. Compare
  NQS to lattice MF and lattice ED (same lattice), never to the continuum directly.
- **Ablation statistics.** Two seeds per configuration; seed spread ≈ 2–8e-4, comparable to
  the differences between MLP/GRU/transformer. The conclusion "backbone doesn't matter" is
  robust; a ranking among the three is not.
- **N = 20 has no exact reference.** Its 0.37 % beyond-Hartree correction is a *prediction*,
  supported by the fact that at N = 8, 12 the same method matched ED to < 0.03 %.
- **Centre of mass.** The spherical model has no centre-of-mass mode (the origin is fixed), so
  its correlation energies are intrinsic. Any comparison to a translation-invariant 3D
  calculation would differ by the CM localisation energy |E|/N — see `HARTREE_CHECK.md`.
- **t-VMC** is a prototype (§11).

---

## 15. Suggested reading order

1. `lattice.py` docstring, then the class — the physics lives here.
2. `meanfield.py` (short) and `exact.py` (short) — the two references.
3. `model.py`: `_forward_logits` and `init_mean_field` — the two ideas that make it work.
4. `vmc.py`: `local_energy`, then `train` — the VMC loop.
5. `run_demo.py` — how it all connects; then `run_ablation.py`.
6. `bogoliubov.py` + `HARTREE_CHECK.md` — why the correction is believable.
7. Dynamics module last; it is a separate thread.

# Neural quantum state for the quantum gravitational soliton

**Key results.** Where exact diagonalisation is possible, the NQS recovers the full
beyond-mean-field (correlation) energy: 100.3 ± 4.3 % at N=8, K=8 and 97.7 ± 9.2 % at
N=12, K=12 (error vs exact < 0.03 %, vs 1.2 % / 0.7 % for Hartree). At N=20, K=16, where
exact diagonalisation is impossible (~3×10⁹ states), it predicts a 0.37 % beyond-Hartree
correction with condensate fraction 0.9989. An ablation over four backbones shows the
transformer is *not* what makes it work: a structured binomial head initialised exactly at
the Hartree state is decisive, and an MLP with 7× fewer parameters matches the transformer.
See `CODE_GUIDE.md` for a file-by-file walkthrough and `HARTREE_CHECK.md` for the validation
against exact diagonalisation and Bogoliubov theory.

`soliton_nqs` finds the many-body **quantum** ground state of N self-gravitating
bosons — the quantum version of the Schrödinger–Poisson (SP) soliton studied in
the AxionSoliton project — with a **transformer neural quantum state (NQS)**
trained by variational Monte Carlo (VMC).  Spherical symmetry is imposed and only
the radial direction is discretised into K = 8–32 modes.  For small (N, K) the
result is checked against exact diagonalisation; the lattice mean-field (Hartree)
soliton is always computed as the classical reference.

## Units and conventions (same as the project's notes / `NormalModes.py`)

ħ = m = 4πG = 1.  Classical limit: `i ∂_t ψ + ½Δψ − Φψ = 0, ΔΦ = |ψ|²`.
Standard soliton: χ₀(0)=1, μ₀ = 25.9, ε₀ = −0.692, r₁/₂ = 1.30 (χ₀² = ½),
total classical energy E_std = ε₀μ₀/3 = −5.97 (virial theorem E = −T).
`chi0list.dat` (copied from `python/`) supplies χ₀(r) for the continuum profile.

## Formulation

**Hamiltonian.** N bosons with pair potential −1/(4π|x−x'|):

    H = Σ_a p_a²/2 − Σ_{a<b} 1/(4π|x_a − x_b|)

**Spherical ansatz + radial modes.** Every particle sits in an ℓ=0 orbital.  The
radial line is divided into K shells with edges 0 = e₀ < … < e_K = R (logarithmic
grid by default), centres r_j, widths D_j; each shell is one bosonic mode b_j.
With u = rψ and a_j = √(4πD_j) u(r_j) the finite-volume kinetic energy is a
tridiagonal one-body matrix (nearest-neighbour hopping, all negative), and the
monopole part of 1/|x−x'| for spherical shells is 1/max(r,r'), so gravity is a
pure density–density coupling W_ij = −1/(4π max(r_i,r_j)).  The lattice
Hamiltonian is a Bose–Hubbard-like chain with long-range attraction:

    H = Σ_ij h_ij b_i†b_j + ½ Σ_ij W_ij (n_i n_j − δ_ij n_i)

Because all off-diagonal elements are negative, the ground state is positive in
the Fock basis (Perron–Frobenius): a real ansatz ψ(n) = √p(n) is exact in
principle and no sign/phase problem exists.

**Grid.** A coarse *uniform* grid lets the lattice model collapse onto the
innermost site (finite-difference kinetic energy is bounded, gravity is not) —
a discretisation artefact.  The default log grid (`r_min = 12/N`,
`r_max = 6 r₁/₂`) avoids it; the lattice Hartree energy then agrees with the
continuum finite-N Hartree energy to ~3 % (K=8), 0.7 % (K=12), 0.1 % (K=16).

**Finite-N classical reference.** A Hartree product state of N particles obeys
the GP equation with coupling (N−1), i.e. it is the standard soliton rescaled by
S_H = (N−1)/μ₀:  E_H = (E_std/μ₀) S_H² N,  r₁/₂ = 1.30/S_H,
ρ_H(r) = N/(N−1) · S_H⁴ χ₀(S_H r)².  The lattice `mean_field()` converges to it.

**Ansatz.**  Autoregressive transformer over the occupation string
n = (n₁,…,n_K), Σn_j = N:

    p(n) = Π_j p(n_j | n_<j),   ψ(n) = √p(n)

Each conditional is a masked categorical over n_j ∈ {0,…,N−Σ_{i<j}n_i}, so every
sample is a valid N-particle state and ψ is exactly normalised.  The default
"poly" head writes the logits as `log C(rem,n) + Σ_m c_m(context)·(n/N)^m·N +
small free residual`, i.e. a binomial (mean-field-like) backbone whose
parameters are produced by the transformer.  This lets the network be
initialised **exactly** at the mean-field state (`init_mean_field`), and keeps
the tails well behaved, which is what makes the local-energy variance small.

**Training.**  E_loc(n) = E_diag(n) + Σ_hops (−t) √(n_i(n_j+1)) ψ(n')/ψ(n) with
only 2(K−1) neighbours per sample; gradient 2⟨(E_loc − Ē) ∂log ψ⟩ with Adam
(minSR / stochastic reconfiguration is available with `--sr`).  Observables:
energy, ⟨n_j⟩ and fluctuations, one-body density matrix ⟨b_i†b_j⟩ →
condensate fraction and natural orbital.

## Soliton growth / evaporation (soliton + gas mode basis)

`soliton_nqs/modes.py` builds the l=0 Schrödinger modes of the soliton potential
in a box (mode 0 = soliton orbital, then excited bound states and box-continuum
"gas modes"), the bare one-body matrix h0 and the four-index gravity tensor U.
`soliton_nqs/dynamics.py` does exact many-body time evolution (sparse H,
expm_multiply) and the classical / truncated-Wigner multimode field in the same
basis.  `run_dynamics.py` compares them:

    python run_dynamics.py --Ns 6 --K 8 --gas 3:2 --tmax 3000   # cold gas
    python run_dynamics.py --Ns 6 --K 8 --gas 7:2 --tmax 3000   # hot gas

Result (see notes): the exact quantum evolution shows net evaporation of the
soliton mode, the phase-averaged classical field only coherent oscillations;
truncated Wigner tracks the quantum mean.

## Soliton formation from a gas

`run_formation.py` starts from N quanta in free-particle box orbitals on the
radial lattice (a Fock-state gas, no soliton) and follows the exact many-body
dynamics; the condensate fraction (largest eigenvalue of <b_i^+ b_j>/N), the
natural orbital and the density show a soliton core forming within ~1 period
(natural orbital overlap 0.97-0.99 with the mean-field soliton) plus a heated gas.

    python run_formation.py --N 8 --K 10 --gas 4,4,0 --tmax 4000

## Time-dependent NQS (t-VMC) prototype

`soliton_nqs/tvmc.py` implements TDVP for the complex transformer ansatz
(minSR form, Heun integrator); `run_tvmc_check.py` validates it against the
exact evolution for the collapse of N=6 quanta from the lowest box orbital.
It tracks the exact dynamics for ~0.6 soliton periods, then loses the growing
inner-shell components (rare-configuration growth) — see notes Sec. 8 for the
remedies (projected t-VMC, momentum/mode basis, larger batch) and the
single-GPU NQS programme.

## Ablation: is the transformer needed?  (run_ablation.py)

Same lattice/head/init/training, only the context network changes.  Result:
the structured head + exact Hartree initialisation is decisive (free softmax
head fails); some context is needed (head alone stays at mean field); MLP, GRU
and transformer are equivalent at K<=16 (all within 1-3e-4 of exact), the MLP
being 3-4x cheaper.  The transformer is kept for large K / dynamics, not
because it wins here.  Options: `TransformerNQS(..., backbone='mlp'|'rnn'|'none'|'transformer')`.

## Files

    soliton_nqs/lattice.py    RadialLattice: grid, h_ij, W_ij, classical references
    soliton_nqs/meanfield.py  lattice Hartree (GP) minimiser
    soliton_nqs/exact.py      exact diagonalisation in the fixed-N Fock basis
    soliton_nqs/model.py      TransformerNQS (autoregressive, masked, poly/free head)
    soliton_nqs/vmc.py        local energy, Adam VMC, minSR, observables
    soliton_nqs/modes.py      soliton + gas mode basis (Schrödinger modes), h0, U_ijkl
    soliton_nqs/dynamics.py   exact many-body dynamics, classical / truncated-Wigner field
    run_demo.py               ground-state demo with plots
    run_dynamics.py           growth/evaporation demo (quantum vs classical)
    run_formation.py          soliton formation from a gas (exact quantum vs classical)
    soliton_nqs/tvmc.py       TDVP / t-VMC for the complex transformer ansatz
    run_tvmc_check.py         t-VMC validation vs exact evolution
    run_ablation.py           backbone ablation (none / MLP / GRU / transformer, poly vs free head)
    soliton_nqs/bogoliubov.py Bogoliubov (BdG) correction around the lattice Hartree state
    check_bogoliubov.py       table: Bogoliubov vs exact correlation energy
    check_hartree.py          zero-mode analysis (projected Bogoliubov, depletion)
    state_character.py        coherent / Fock / squeezed character of the ground state (ED)
    CODE_GUIDE.md             how the code is structured and how each piece works
    HARTREE_CHECK.md          is Hartree good enough, and does the NQS go beyond it?
    results/                  demo outputs (json, png)

## Running

    pip install torch numpy scipy matplotlib
    python run_demo.py --N 8  --K 8  --steps 500 --batch 1024          # ED check (6435 states)
    python run_demo.py --N 12 --K 12 --steps 500 --batch 1024          # ED check (92378 states)
    python run_demo.py --N 20 --K 16 --steps 500 --batch 1024 --no-ed  # NQS vs mean field
    python run_demo.py --N 26 --K 32 --steps 1000 --batch 2048 --no-ed --d_model 64

Everything runs on CPU in minutes for K ≤ 16; a GPU makes K = 32 / larger N
comfortable.  Options: `--grid uniform`, `--r_min`, `--r_max`, `--head free`,
`--sr`, `--d_model/--layers/--heads`.

## Caveats / next steps

* Restricting every particle to ℓ=0 orbitals is an approximation to the true
  L=0 ground state (pairs in (ℓ,ℓ) coupled to L=0 are excluded); the radial
  lattice adds a discretisation error controlled by K and the grid.
* Quantum corrections to the soliton are O(1/N): at N = 8 the exact energy is
  ~1.3 % below the lattice Hartree energy; the NQS resolves this to a few 10⁻⁴.
* Natural extensions: complex phase head (`use_phase=True`) for excited/dynamical
  states, non-zero ℓ modes, self-interaction (λφ⁴) term added to W, larger N with
  a GPU, and time evolution (t-VMC) of soliton growth/evaporation.

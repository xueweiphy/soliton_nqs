# Is the Hartree approximation good enough, and does the NQS go beyond it?

Three independent checks. All numbers are on the *same lattice* in each row, so the
comparison is exact — no continuum extrapolation enters.

## 1. Hartree is badly wrong at small N (3D, exactly solvable)

For N = 2 the many-body problem is exactly solvable: in the zero-total-momentum sector the
pair reduces to one particle of mass 1/2 in the potential -1/(4 pi r), which is a
hydrogen-like problem (continuum answer E = -1/(64 pi^2) = -1.5831e-3). Solving it on the
same 3D lattice used by the NQS (3D-lattice code kept in a separate, unpublished repo):

| L | dx | box / a_B | E_Hartree | E_exact | Hartree / exact |
|---|---|---|---|---|---|
| 24 | 6.28 | 6 | -1.0418e-3 | -1.7153e-3 | 0.607 |
| 32 | 6.28 | 8 | -8.0699e-4 | -1.5879e-3 | 0.508 |
| 32 | 9.42 | 12 | -6.8940e-4 | -1.5357e-3 | **0.449** |

Mean field recovers less than half the binding energy — a factor 2.2 error — converging to
the continuum ratio 0.434. Hartree is not a controlled approximation at small N.

## 2. The NQS recovers essentially all of the correlation energy (vs exact diagonalisation)

Radial (spherical) model, where exact diagonalisation in the fixed-N Fock basis is possible:

| N | K | E_Hartree | E_exact | correlation energy | NQS | fraction captured |
|---|---|---|---|---|---|---|
| 8 | 8 | -0.138920 | -0.140565 | -1.645e-3 | -0.140570 +- 7e-5 | **100.3% +- 4.3%** |
| 12 | 12 | -0.502720 | -0.506423 | -3.703e-3 | -0.506336 +- 3.4e-4 | **97.7% +- 9.2%** |

The Hartree error is 1.2% of E at N = 8 and 0.73% at N = 12; the NQS error against exact is
below 0.03% in both. So where it can be checked, the ansatz captures the beyond-Hartree
physics, not a fraction of it.

## 3. Bogoliubov confirms the size of the effect independently

The standard leading beyond-mean-field theory (BdG around the Hartree orbital,
`soliton_nqs/bogoliubov.py`) gives, for the same systems:

| N | K | correlation energy (exact) | Bogoliubov | Bog / exact |
|---|---|---|---|---|
| 6 | 10 | -8.372e-4 | -7.243e-4 | 0.865 |
| 8 | 8 | -1.645e-3 | -1.480e-3 | 0.900 |
| 12 | 12 | -3.703e-3 | -3.447e-3 | 0.931 |

Bogoliubov approaches the exact answer from below as N grows, exactly as a leading-order
1/N theory should. So the correlation energy the NQS finds is the standard beyond-mean-field
effect, at the size analytic theory predicts — it is not a numerical artefact.

## 4. The zero mode shows up here too — and it argues *for* the NQS

The Bogoliubov spectrum at N = 8 is
`omega = 1.609, 0.715, 0.365, 0.146, 0.062, 0.045, 0.0221`.
The softest mode is a factor ~2 below its neighbour and is the partner of the U(1) phase zero
mode, i.e. the direction d(chi)/dN in which the soliton changes size when a particle is added
— the same mode that makes "occupation of mode 0" a bad soliton diagnostic.

Its effect on the naive BdG numbers is large:

| quantity | naive BdG (project out phi only) | also project out d(phi)/dN | exact |
|---|---|---|---|
| depletion 1 - lambda/N (N=8) | 0.0492 | 0.0009 | **0.0064** |
| correlation energy | -1.480e-3 (90% of exact) | -1.106e-4 (7%) | -1.645e-3 |

Naive Bogoliubov overestimates the depletion by a factor of 8 and projecting the mode out
entirely destroys the energy: the mode needs the proper number-conserving (Castin-Dum)
treatment, in which it contributes phase diffusion rather than depletion. **The NQS has no
such ambiguity** — it works at strictly fixed N by construction, so the zero-mode pair never
enters. That is a real argument for the method, not just a convenience.

## Caveats to state before quoting any of this

* **Centre of mass.** For a self-bound system the Hartree state pays a spurious CM
  localisation energy, `E_CM = T/N = |E|/N` — a *known mean-field artefact*, not interesting
  correlation. The spherical model has no CM mode (the origin is fixed), so its numbers above
  are pure correlation. The 3D ansatz pins the soliton with a localised orbital, so it also
  computes the intrinsic energy; this must be said explicitly, because a naive comparison
  with a translation-invariant "exact" calculation would differ by |E|/N.
* **The 3D correlation energy is not yet validated at the level of the spherical one.**
  The 3D runs give `N x dE/|E| ~ -0.03` against `-0.09` in the spherical model. That gap is
  either physical (different discretisation and dimensionality) or an ansatz limitation (the
  two-body Jastrow reaches only 2 cells). The next check is Bogoliubov *in 3D* as the anchor,
  plus a Jastrow-range scan.

## Bottom line

Hartree is exact only as N -> infinity: it loses a factor 2 at N = 2 and ~1% of the energy at
N = 8-12, and it has a well-known 1/N artefact from centre-of-mass localisation. The NQS
reproduces the exact correlation energy to within its error bars wherever exact
diagonalisation is available, and agrees with Bogoliubov at the level Bogoliubov is expected
to be right. What still needs doing is the same validation *in 3D*: Bogoliubov on the 32^3
lattice as an independent anchor, and a Jastrow-range scan to confirm the ansatz is converged.

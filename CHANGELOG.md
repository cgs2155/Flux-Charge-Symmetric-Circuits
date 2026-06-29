# Changelog

All notable changes to **fluxcharge** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-06-29

Interactive spectroscopy, partial duality, and a substantial multi-mode
**correctness** pass (honest guards instead of silently-wrong numbers).

### Added
- **Interactive spectrum explorer** (`fluxcharge/interactive.py`), built on
  `matplotlib.widgets` and the package's own numerics, so it works for the
  gyrator / quantum-phase-slip circuits scqubits cannot represent:
  - `spectrum_slider` — energy levels vs. live parameter sliders.
  - `spectrum_vs_param` — levels or transitions as curves vs. one swept
    parameter, with sliders for the rest, bias-aware sweep ranges (flux `0..2π`,
    charge `0..1`), optional transition-matrix-element colouring, and a
    debounced update loop.
  - `auto_cutoffs` — basis-aware default truncations (charge basis for periodic
    modes, flux/Fock for extended), capped for multi-mode responsiveness.
- **In-app "Live" view** — `spectrum_vs_param` embedded in the desktop app via
  `FigureCanvasTkAgg`, sweeping an **external Noether variable** (a transmon's
  gate charge, a fluxonium's loop flux); offers to add a bias if the circuit has
  none, and lets you choose among several.
- **Partial-dual transformation** `move_across_gyrator` (manuscript Sec.
  "Partial Dual Transformations"): relocates a reciprocal block across the
  gyrator terminating its port (single-element Tellegen move and multi-element
  parallel→series block), with external offset-charge **bias carry-over**.
- **Exact Williamson spectra for linear multi-mode circuits**: a purely
  quadratic circuit with a dense bracket (e.g. gyrator-coupled oscillators) is
  solved from its symplectic normal-mode frequencies (`symplectic_eigenvalues`,
  wired into `eigenenergies`).

### Changed
- `commutators()` now reports the **full** `i·ħ·(f⁻¹)` — every nonzero bracket,
  including the cross-brackets of a dense multi-mode form — instead of the
  per-pair shortcut. Single-mode results are unchanged (one `±iħ` per pair).
- `canonical()` now also transforms `symplectic_matrix` (`f → S⁻¹fS⁻¹`) when it
  rescales a charge, so `f⁻¹` (commutators) and the Williamson spectrum stay
  consistent with the rescaled coordinates.
- `library.zero_pi` uses a manifest-compact node frame (cleaner planar schematic,
  correct mode classification: one PERIODIC + two EXTENDED).
- Desktop GUI: Numerical diagonalization now shares the bottom row with the
  details report (no scroll box); **Sweep** (static) and **Live** (interactive)
  buttons; per-mode-type dialog on Diagonalize with auto-filled defaults.
- Schematic: parallel sibling elements (e.g. a junction and its junction
  capacitance) are drawn as compact outward "rungs" rather than full-length
  parallels, so multi-element edges stay planar.
- Finalized licensing metadata (authors, repository URL); project version 0.2.0.

### Fixed
- **Multi-mode quantization is no longer silently wrong.** When the reduced
  symplectic bracket is not block-diagonal in the conjugate pairs (a dense
  flux↔charge block, as for 0-π), the per-pair operator basis would drop the
  cross-brackets; the numeric layer now raises `CompactLatticeError` rather than
  return an unjustified spectrum. Block-diagonal circuits (every single-mode one)
  are unaffected and remain verified to machine precision.
- Reduction no longer hangs on gyrator + nonlinear circuits: constraint
  elimination avoids transcendental targets, and a circular/transcendental
  self-consistency now raises `ReductionError` instead of building an infinitely
  nested expression.
- Fixed a cyclic-coordinate over-drop that deleted gyrator-induced dynamics
  (a gyrator terminated by a capacitor is an LC oscillator, not nothing).

### Known limitations
- The numeric spectrum of a **nonlinear multi-mode** circuit (the 0-π qubit and
  the like) is guarded, not computed — use `scqubits.ZeroPi`, a grid, or the
  QuTiP operator export. General multi-mode canonicalization (a real Darboux step
  plus an integer lattice for compact modes) is the open frontier; the verified
  building blocks live in `fluxcharge/canonicalize.py`.

## [0.1.0] — 2026-06-18

Initial release: the symbolic engine and its surrounding tooling.

### Added
- Flux-charge symmetric **LCG formalism** → analytical Hamiltonian for
  lumped-element superconducting circuits (capacitors, inductors, Josephson
  junctions, quantum phase slips, gyrators), via the constraint reduction of
  Salcedo, Cocquyt, Osborne & Houck — reproducing the manuscript's **circulator**
  example exactly.
- **Circuit duality** (`dual`) and **automatic loop inference** (declaring
  planar faces is optional).
- **Numerical diagonalization**, mode-type detection (EXTENDED / PERIODIC /
  DUAL_PERIODIC / FREE), spectrum & wavefunction plotting, parameter sweeps,
  matrix elements and coherence estimates.
- **External flux bias** and **offset charge** (the manuscript's nonzero Noether
  constants); physical units (fF / nH / GHz → spectrum in GHz).
- A validated **circuit library** (transmon, Cooper-pair box, fluxonium, LC,
  phase-slip qubit, gyrator circulator, 0-π) and a guided tutorial.
- **scqubits** YAML import and **QuTiP** operator export.
- A themed desktop **GUI** (`fluxcharge-gui`), a **CLI** (`fluxcharge`), and a
  PyInstaller-packaged app.

[0.2.0]: https://github.com/cgs2155/Flux-Charge-Symmetric-Circuits/releases/tag/v0.2.0
[0.1.0]: https://github.com/cgs2155/Flux-Charge-Symmetric-Circuits/commit/9225048

# fluxcharge — project brief for Claude Code

Symbolic package that turns a lumped-element superconducting circuit (C, L,
Josephson junctions, quantum phase slips, gyrators) into its analytical
Hamiltonian using the **flux-charge symmetric LCG formalism** of:

> C. Salcedo, S. Cocquyt, A. Osborne, A. A. Houck,
> "Gyrators for superconducting circuit design."

The user is a co-author. **Fidelity to the manuscript and honesty about what is
automated vs. assumed matter more than convenience.** Never silently return a
wrong or incomplete Hamiltonian.

## How to work in this repo
- Source of truth: `fluxcharge/`. Tests: `tests/test_example.py` (run `pytest -q`; 74 tests, all passing).
- Reproduce the manuscript's worked example: `python examples/reproduce_manuscript.py` (asserts `H_package - H_published == 0`).
- CLI: `fluxcharge circuit.txt` (also `--dual`, `-o out.png`, `--raw`, `--no-draw`). GUI: `fluxcharge-gui`.
- Core deps (all required): sympy, schemdraw, matplotlib, networkx. Optional: numpy (numeric), pytest (dev).
- Manuscript sources live outside this repo (main_3_.tex, CircuitExample.tex); keep new behavior consistent with them.

## Package layout
- `elements.py` — Edge + element classes. Capacitor(Q²/2C), Inductor(Φ²/2L),
  JosephsonJunction(−E_J cos Φ, *inductive*), QuantumPhaseSlip(−E_S cos Q,
  *capacitive*), Gyrator(ordered edge pair, ratio G). Convention G₀=1, ħ=1.
- `circuit.py` — `Circuit`: add_capacitor/inductor/josephson/qps/gyrator/loop;
  `connect_to_ground(*nodes)` merges nodes into one ground node (so a dangling
  stub becomes a real shunt) and re-checks planarity; `hamiltonian()` warns on
  floating (degree-1) terminals that would silently drop out;
  builds incidence A (A[e,head]=+1, A[e,tail]=−1), loop matrix B (B[l,e]=±1),
  connection M=½B(P_C−P_I)A, antisymmetric Ω, Lagrangian L=½⟨x|Ω|ẋ⟩−E.
  `hamiltonian(ground=None, open_loops=None, keep=, eliminate=, strict=True, canonical=False)`.
  `validate()` checks Kirchhoff exactness B·A=0. `schematic(...)`, `to_networkx()`.
- `reduction.py` — `Reducer` + `ReductionResult`. Implements the manuscript's
  constraint taxonomy: null-vector constraints (X∈ker Ω, ⟨X|∇E⟩≠0) and Noether
  constraints (X a symmetry of E, not in ker Ω). Gauge = ker(A⊕Bᵀ): global-flux
  (ground a node) and global-charge (open a loop). Reduced symplectic form
  `f = −Ω/2`; completeness = f non-degenerate. `ReductionResult.canonical()`,
  `.commutators()` (= −iħ/coeff per pair; calibrated so transmon gives [φ,q]=iħ),
  `.compact_coordinates()` (coords inside a cos → may live on S¹), `.report()`.
  Constraint elimination prefers a *polynomial* (linear) target over one buried
  in a JJ/QPS cosine, so a working JJ circuit and its QPS dual reduce
  symmetrically (don't pick a transcendental target → no `atan(…√…)` monster).
  When a gyrator couples a *nonlinear* element into a circular/transcendental
  self-consistency `x=f(sin x)` (ill-posed or non-reciprocal-nonlinear), it
  raises **`ReductionError`** promptly instead of hanging on nested `sin(sin(…))`
  or crashing out of `sp.solve` — the honest guard, like `CompactLatticeError`.
- `netlist.py` — `parse_netlist`/`from_netlist` (text format) and `to_netlist`
  (serialize back). Format: `TYPE name n1 n2 [val]` (C/L/J/QPS), `gyrator e1 a b
  e2 a b [ratio]`, `loop name +e1 -e2 ...`, `ground node`, `open loop`, `title`.
- `transformations.py` — `dual(circuit)`: the LCG duality transform
  (φ↔q, A↔Bᵀ so vertices↔faces, C↔L, JJ↔QPS value-preserved, G→−1/G). Involution.
- `observables.py` — `current(result, edge)` / `voltage(result, edge|a,b)` (also
  `ReductionResult.current`/`.voltage`): branch current/voltage operators in the
  reduced coordinates, from a **circuit solve** (constitutive laws + Kirchhoff),
  NOT Heisenberg brackets. Why not `{X,H}`: the reduction eliminates coordinates
  and `canonical()` rescales survivors, so `{φ_v,H}` is *not* the physical node
  voltage once a gyrator mixes the flux/charge sectors (it disagrees with the
  definitional `V=Q/C`). `_solve_circuit` reads branch fluxes/charges off the
  reduced state, then: inductive **current** = `∂E/∂Φ` (Φ/L, E_J sinΦ), capacitive
  **voltage** = `∂E/∂Q` (Q/C, E_S sinQ); node potentials from the capacitor
  voltages (`u_head−u_tail=∂E/∂Q`); **gyrator** half-edge current `I₁=−G V₂`,
  `I₂=+G V₁` (ideal-gyrator relation); capacitive (displacement) currents from
  KCL. Biases enter via the branch-variable offsets. Verified: **KCL closes at
  every node and KVL around every loop, including the non-reciprocal circulator**
  (gyrator half-edges carry current). `voltage(a,b)=u_a−u_b`. Feed the operator
  to `matrix_elements` (now accepts any expression) for numeric values. Natural
  units (ħ=1, G₀=1). Raises only if a node has no capacitive path to the
  reference (a purely-inductive island).
- `schematic.py` — planar schemdraw drawing. Half-gyrator crescent (normal to
  wire, faces partner). QPS = box with center line **normal to the wire**.
  Outer face auto-detected deterministically (largest face, fewest gyrator edges,
  then lexical) — independent of loop declaration order and of the gauge.
- `gui.py` — themed Tkinter app (`fluxcharge-gui`). `compute()` is the headless
  core (testable). Renders schematic + Ĥ + commutators via matplotlib mathtext
  (cm fontset); operators are hatted, c-number params are not. "LaTeX (system
  TeX)" toggle (usetex with mathtext fallback). "Dualize" button + a partial-dual
  control ("move edges: … across gyrator ⤨", `move_across_gyrator`). "Sweep" (static
  spectrum-vs-parameter) and **"Live"** (interactive `interactive.spectrum_vs_param`
  embedded via `FigureCanvasTkAgg`, so the sliders are responsive in-app).
- `interactive.py` — standalone interactive spectrum explorers on `matplotlib.
  widgets.Slider` + the package's own `eigenenergies` (so they work for the
  gyrator/QPS circuits scqubits can't represent; scqubits' own widgets are
  Jupyter/ipywidgets-only). `spectrum_slider` (levels as rows) and
  `spectrum_vs_param` (levels/transitions as curves vs one swept param; bias-aware
  ranges — flux 0..2π, charge 0..1; auto-picks a bias axis; optional `weight_by`
  matrix-element colouring). Accept `fig=` to embed in a Tk-bound figure.
- `__main__.py` — CLI (`analyze`, `main`).

## Key validated results (keep these passing)
- Circulator reproduces published H exactly; complete; [φ_v2,q_f3]=iħ.
- Transmon q²/2C−E_J cosφ ([φ,q]=iħ, φ compact); fluxonium +φ²/2L; QPS‖L
  (charge is the compact/cos variable, [φ,q]=−iħ); parallel caps canonical
  H=φ²/2L+q²/2(C1+C2).

## Conventions / guardrails to preserve
- **One loop (and one node) is always redundant.** The reduction must complete
  for ANY `open`/`ground` choice or none — it removes residual redundant
  *linear combinations* via `Reducer._complete_reduction()` (null directions of
  the reduced symplectic form that are absent from the energy). Don't reintroduce
  a dependence of completeness on the gauge choice. A null direction that *does*
  appear in the energy is a real singularity → leave incomplete so strict=True raises.
- The schematic's outer face is a property of the planar embedding, NOT the
  gauge; never couple `open_loops` to the drawing's `outer_loop`.
- Commutator sign is calibrated to the manuscript ([φ,q]=iħ for the transmon).
- Quote/refer to the manuscript faithfully; surface the S¹-compactness caveat.

## External bias (implemented)
- `circuit.set_flux_bias(loop, value=None)` threads an external flux through a
  loop; `set_offset_charge(node, value=None)` puts an offset/gate charge on a
  node (the LCG dual). Both are the manuscript's nonzero Noether constants,
  injected as constant offsets to the edge fluxes/charges in `Circuit.energy()`
  (split evenly over the loop's inductive / node's capacitive edges so the
  symbol equals the *physical* bias: flux period 2π with the sweet spot at π,
  charge period 1). Netlist directives `flux <loop> [val]` / `offset <node>
  [val]`. `dual()` carries them across the loop↔node swap. They flow as ordinary
  parameters into the symbolic H and the numerics (sweep over them like any param).

## Automatic loop inference (implemented)
- Declaring loops is optional. `Circuit.infer_loops()` (called by `hamiltonian()`
  when no loops exist) derives them from the graph: `topology.py` subdivides each
  edge with a midpoint, finds a planar embedding via `networkx.check_planarity`,
  and traces the **faces** (so `dual`/`schematic` work). A non-planar circuit
  falls back to a tree-based fundamental cycle basis (Hamiltonian only) and warns;
  `Circuit._planar` records which. Inferred loops reproduce the hand-declared
  transmon/fluxonium/circulator spectra (tested).

## Known issues / open problems (IMPORTANT — read before trusting multi-mode output)
- **UPDATE (numeric guard wired in; 0-π now GUARDS, honestly).** The numeric layer
  now checks whether the reduced bracket is **block-diagonal in the conjugate
  pairs** (`numerics.bracket_is_block_diagonal`). Block-diagonal ⇒ the per-pair
  operator basis is exact ⇒ existing path (every single-mode circuit: transmon,
  fluxonium, QPS, the circulator — verified to machine precision). **Dense**
  bracket ⇒ per-pair would silently drop cross-brackets ⇒ `_OperatorBuilder`
  raises `CompactLatticeError`. Two carve-outs: a purely **quadratic** dense
  circuit (linear multi-mode, e.g. gyrator-coupled oscillators) is solved exactly
  by `eigenenergies` via `canonicalize.symplectic_eigenvalues` (Williamson normal
  modes, validated: LC → 1/√(LC)); and the symbolic H is always correct.
  Consequence: **`library.zero_pi` is dense + nonlinear, so its numeric spectrum
  now RAISES** — the earlier "manifest frame diagonalizes cleanly / [0,0.194,…]"
  claim was the per-pair build silently dropping a ~0.67 cross-bracket and was
  *unjustified* (the manifest frame fixed the mode *classification*, not the
  dense canonicalization). For 0-π's number: scqubits.ZeroPi / a grid / QuTiP
  export. `commutators()` now reports the **full** `i·ħ·(f⁻¹)` (all nonzero
  brackets), so a dense circuit shows its real cross-brackets (e.g. 0-π's
  `[φ_v3,q_f1]=2iħ/3`, while the naive per-pair partner `[φ_v3,q_f3]=0`); single-
  mode is unchanged (one `±iħ`). `canonical()` now also updates
  `symplectic_matrix` (`f→S⁻¹fS⁻¹`) so `f⁻¹` and the Williamson path stay
  consistent with the rescaled coordinates.
- **Multi-mode canonicalization is wrong when the reduced symplectic form is not
  block-diagonal.** `reduction._pairs_from_form` + `ReductionResult.canonical()`
  read each conjugate pair off a single entry of the reduced form (per graph
  object) and rescale per-pair. This is correct only when the flux↔charge block
  of `f⁻¹` is diagonal — i.e. every *single-mode* circuit (transmon, fluxonium,
  the single-mode circulator), which is why those validate to machine precision.
  For a genuinely multi-mode circuit the flux↔charge block is **dense** (each
  flux brackets several charges), the off-diagonal brackets are silently
  discarded, and the reported commutators **and spectrum are wrong**. Confirmed
  on the **0-π qubit**: package `canonical()` gives gaps `[0,0.702,1.060,…]`
  vs the symplectically-correct `[0,0.641,1.000,…]` (E_J=C=L=C_J=1), the correct
  value cross-checked against an independent full-bracket diagonalizer that
  matches fluxonium to 1e-15. The fix is a real symplectic (Darboux) transform
  before pairing; for the block-antidiagonal case it is `q' = (Bᵀ)⁻¹q`. Design
  notes: `~/Downloads/CANONICALIZATION_FIX.md` and `zeropi_canonicalization.pdf`.
- **The dual of a multi-mode circuit is therefore unreliable.** `dual(zero_pi)`
  is mis-canonicalized the same way, so its spectrum does not match the original
  (duality *must* preserve the spectrum — verified for the single-mode
  transmon↔QPS to 1e-15 — so the mismatch is the bug above, not the transform
  per se, but this is unproven for multi-mode until canonicalization is fixed).
- **Multi-mode non-reciprocal (gyrator) quantization is not thought through.**
  A gyrator can put nonzero entries in the flux–flux / charge–charge blocks of
  the reduced form, so even the general Darboux step (not just the
  block-antidiagonal special case) is needed; this is untested and unhandled.
- **Compactness/periodicity is a second, separate open problem.** Mode
  classification (`numerics.classify_modes`) is per graph-pair and can't see a
  compact normal-mode combination hidden in coordinate mixing (e.g. 0-π's θ has
  no quadratic term in the φ/θ frame but every node flux carries one). A real
  symplectic transform lives in `Sp(2n,ℝ)` but a compact coordinate's integer
  (Cooper-pair / fluxoid) lattice is preserved only by `Sp(2n,ℤ)`; a generic
  Darboux frame can rotate a periodic coordinate off its lattice (→ `cos(q/2)`),
  which has no integer-basis representation. 0-π evades this (coupling is purely
  flux↔charge, all-extended) but **its dual does not**. Plan: classify modes
  *after* canonicalization, integer basis for compact modes, and a guard that
  raises rather than silently using an oscillator basis for a compact mode.
- Single-mode results and the manuscript's circulator example are unaffected and
  remain correct.

### Progress (verified) and what remains
- **Built + verified** in `fluxcharge/canonicalize.py` (tests T1/T2/T3):
  `canonical_from_bracket` (general Darboux, handles gyrator-populated flux-flux/
  charge-charge blocks — reproduces the exact non-reciprocal oracle
  `{0.6431, 2.0094}`), `symplectic_eigenvalues` (convention-free Williamson
  oracle), and `compact_flux_modes` (compactness from the gauge lattice:
  transmon 1, fluxonium 0, 0-π 1 = θ).
- **Decisively confirmed** (independent grid/charge diagonalization of the
  textbook 0-π): θ MUST be compact. With θ in the integer basis the spectrum is
  `[0, 2.361, 2.712, 3.179, 5.175, 5.254]` and stable; treating θ as extended
  (the package's current all-oscillator path) **diverges** (gaps → 0 as the box
  grows). The earlier "converged" all-extended `[0,0.641,…]` was an oscillator-
  localization artifact, i.e. **wrong**. So the 0-π fix needs BOTH the symplectic
  canonicalization AND the compact (integer) basis for θ.
- **scqubits cross-validation (built):** `interop.to_scqubits_yaml` +
  `cross_check_spectrum`. Finding: scqubits' *general* `Circuit` class is a clean
  oracle only for charge networks (transmon 1e-13); for inductive circuits it
  disagrees with grids AND scqubits' own predefined classes -- fluxcharge matches
  the grid/predefined classes, `Circuit` is the outlier. Use grids for inductive
  validation.
- **Compact-frame guard (DONE):** `numerics.check_compact_frame` detects a
  compact mode hidden by coordinate mixing (the inductive/capacitive quadratic
  form is rank-deficient with no coordinate-aligned unconfined direction) and
  raises `CompactLatticeError` instead of a silently-wrong spectrum. **0-π now
  raises** (honest) rather than printing a meaningless number; single-mode
  circuits and the circulator are unaffected. KEY obstruction, verified: the
  natural compact direction θ = φ_n2+φ_n3 gives `cos(θ/2)` (a half-integer cosine
  with no integer-lattice representation), so even "the user says θ is compact"
  is insufficient -- a lattice-aware FRAME (integer cosines; the
  centered/checkerboard lattice) is required.
- **Decision (division of labor):** fluxcharge does NOT auto-solve the compact
  multi-mode spectrum. Experiment confirmed 0-π needs its full identification
  lattice (the centered/checkerboard lattice) -- naive per-mode periodic grids
  give spuriously degenerate, wrong spectra; this is research-grade and is what
  `scqubits.ZeroPi` exists to handle. So for the NUMERIC spectrum of a compact
  multi-mode *reciprocal* circuit, fluxcharge guards (raises `CompactLatticeError`
  pointing to scqubits.ZeroPi / a grid / its QuTiP operator export) rather than
  returning a wrong number. fluxcharge's symbolic Hamiltonian for these circuits
  is still correct and is the deliverable; its *unique* numeric value is the
  single-mode + gyrator/QPS circuits scqubits cannot represent. The
  canonicalizer / symplectic oracle / compactness classifier in
  `canonicalize.py` remain available for building a lattice-aware path later, but
  that (and the non-reciprocal compact-compact / Harper sector) is deferred.

## Deferred / possible next steps (v0.2)
- Partial-dual transformation, gyrator series/parallel + open/closed-terminated
  deletion rules, cascade-to-transformer, NCG reducibility (the paper's Sec.
  "Transformations").
- A scqubits-YAML importer (now that loops are auto-inferred, their reciprocal
  circuits become drop-in) + cross-validation of spectra.
- A true drag-and-drop canvas builder (the face-detection hook is now
  `infer_loops`; `to_networkx()` + `schematic(positions=)` remain the layout hooks).
- LICENSE + CITATION.cff are filled in; only the DOIs (Zenodo archive + paper)
  remain as placeholders, pending the manuscript being posted.

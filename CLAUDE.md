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
- Source of truth: `fluxcharge/`. Tests: `tests/test_example.py` (run `pytest -q`; 17 tests, all passing).
- Reproduce the manuscript's worked example: `python examples/reproduce_manuscript.py` (asserts `H_package - H_published == 0`).
- CLI: `fluxcharge circuit.txt` (also `--dual`, `-o out.png`, `--raw`, `--no-draw`). GUI: `fluxcharge-gui`.
- Core deps (all required): sympy, schemdraw, matplotlib, networkx. Optional: numpy (numeric), pytest (dev).
- Manuscript sources live outside this repo (main_3_.tex, CircuitExample.tex); keep new behavior consistent with them.

## Package layout
- `elements.py` — Edge + element classes. Capacitor(Q²/2C), Inductor(Φ²/2L),
  JosephsonJunction(−E_J cos Φ, *inductive*), QuantumPhaseSlip(−E_S cos Q,
  *capacitive*), Gyrator(ordered edge pair, ratio G). Convention G₀=1, ħ=1.
- `circuit.py` — `Circuit`: add_capacitor/inductor/josephson/qps/gyrator/loop;
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
- `netlist.py` — `parse_netlist`/`from_netlist` (text format) and `to_netlist`
  (serialize back). Format: `TYPE name n1 n2 [val]` (C/L/J/QPS), `gyrator e1 a b
  e2 a b [ratio]`, `loop name +e1 -e2 ...`, `ground node`, `open loop`, `title`.
- `transformations.py` — `dual(circuit)`: the LCG duality transform
  (φ↔q, A↔Bᵀ so vertices↔faces, C↔L, JJ↔QPS value-preserved, G→−1/G). Involution.
- `schematic.py` — planar schemdraw drawing. Half-gyrator crescent (normal to
  wire, faces partner). QPS = box with center line **normal to the wire**.
  Outer face auto-detected deterministically (largest face, fewest gyrator edges,
  then lexical) — independent of loop declaration order and of the gauge.
- `gui.py` — themed Tkinter app (`fluxcharge-gui`). `compute()` is the headless
  core (testable). Renders schematic + Ĥ + commutators via matplotlib mathtext
  (cm fontset); operators are hatted, c-number params are not. "LaTeX (system
  TeX)" toggle (usetex with mathtext fallback). "Dualize" button.
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

## Deferred / possible next steps (v0.2)
- Partial-dual transformation, gyrator series/parallel + open/closed-terminated
  deletion rules, cascade-to-transformer, NCG reducibility (the paper's Sec.
  "Transformations").
- A true drag-and-drop canvas builder; the hard piece is automatic face/loop
  detection from a drawn planar layout (build a rotation system from node
  positions, trace faces). `to_networkx()` + `schematic(positions=)` are the hooks.
- LICENSE + CITATION.cff still have placeholders to finalize before release.

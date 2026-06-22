# fluxcharge

Symbolic Hamiltonians for lumped-element superconducting circuits, in the
**flux–charge symmetric** formalism of

> *Gyrators for superconducting circuit design*
> C. Salcedo, S. Cocquyt, A. Osborne, A. A. Houck.

You describe a circuit as an oriented graph of **inductors, capacitors,
Josephson junctions, quantum phase slips and gyrators**, declare its loops, and
the package builds the boundary maps, the connection matrix `M`, the
antisymmetric form `Ω`, the Lagrangian, and finally an **analytical
Hamiltonian** as a SymPy expression.

The construction reproduces the manuscript's worked non-reciprocal example
(`H = (Gφ + q)²/2C + q²/2C − E_J cos φ`) exactly, and the textbook LC
oscillator, both as automated tests.

---

## Dependencies

| package        | required | purpose                                          |
|----------------|----------|--------------------------------------------------|
| Python         | ≥ 3.9    | —                                                |
| **sympy**      | yes      | all symbolic algebra (matrices, calculus, trig)  |
| **schemdraw**  | yes      | lumped-element schematic drawing                 |
| **matplotlib** | yes      | rendering backend for schematics (schemdraw uses it) |
| **networkx**   | yes      | planar layout / topology graph                   |
| numpy          | optional | numerical evaluation of the symbolic results     |
| pytest         | dev only | running the test suite                           |

The symbolic core only needs **SymPy**; the drawing features pull in
**schemdraw**, **matplotlib** and **networkx**, which are installed by default.
Everything the package returns is a SymPy object, so you can `lambdify`,
substitute numbers, differentiate, or hand it to QuTiP/`scipy` yourself.

---

## Install

```bash
# from the project root (the folder containing pyproject.toml)
pip install -e .
```

or, without installing, just put the folder on your path:

```bash
export PYTHONPATH=/path/to/fluxcharge-project
```

---

## Workflow: a circuit from a text file

The fastest way in is a **netlist** — a short text file describing the circuit.
Running it produces the schematic, the Lagrangian and the Hamiltonian in one go:

```bash
fluxcharge mycircuit.txt            # or:  python -m fluxcharge mycircuit.txt
```

A netlist for the manuscript circulator (`examples/circulator.txt`):

```
title Circulator

# two-terminal elements:  TYPE  name  node1 node2  value
#   C capacitor   L inductor   J Josephson junction   QPS quantum phase slip
J    e1  v1 v2  E_J
C    e2  v2 v3  C
C    e3  v3 v1  C

# gyrator:  gyrator  edge1 n1a n1b   edge2 n2a n2b   ratio
gyrator  e4 v1 v3   e5 v2 v3   G

# loops (faces of the planar circuit), as signed edge lists
loop  f1  +e3 +e4
loop  f2  +e1 -e4 +e5
loop  f3  +e2 -e5
loop  f4  -e1 -e2 -e3

# gauge (optional): a ground node and the open / outer loop
ground v1
open   f4
```

Blank lines and `#` comments are ignored. A value shared by two elements (the
two `C`s above) becomes a shared symbol; omit the value for a unique one. The
command prints the summary, Lagrangian, reduction report and (canonical)
Hamiltonian, and writes `mycircuit.png`. Flags: `-o out.png`, `--no-draw`,
`--raw` (skip canonicalization), `--no-lagrangian`. Add
`--param E_J=15 --param C=1` (repeatable, or comma-separated) to also classify
the modes and report eigenenergies, `--levels N` to choose how many, and
`--wavefunctions out.png` to save a potential/wavefunction plot.

From Python:

```python
from fluxcharge import from_netlist
ckt = from_netlist("mycircuit.txt")        # or a netlist string
result = ckt.hamiltonian(ground=ckt.ground, open_loops=ckt.open_loops)
ckt.schematic(path="mycircuit.png")
```

### Desktop app

For a point-and-click version, launch the bundled UI:

```bash
fluxcharge-gui            # or:  python -m fluxcharge.gui
```

Build the circuit in the netlist panel — by typing, or with the *Add
element / gyrator / loop* helpers so you needn't know the syntax — then press
**Generate** to draw the schematic and show the Lagrangian, the Hamiltonian and
the canonical commutation relations. The Hamiltonian and brackets are typeset as
operators (`\hat H`, `\hat\phi`, `\hat q`, with c-number parameters left plain).
Tick **E_C, E_L, n̂** to re-present the Hamiltonian and commutators in the
familiar qubit units: the conjugate charge `q` is relabelled to the Cooper-pair
number `n`, each capacitance becomes a charging energy `E_C = 1/(8C)` and each
inductance an inductive energy `E_L = 1/L` (so a transmon reads
`4 E_C n² − E_J cos φ` and a fluxonium `4 E_C n² + E_L φ²/2 − E_J cos φ`), with
the definitions shown beneath. Rendering uses matplotlib's mathtext by default;
tick **LaTeX (system TeX)** to render through a real LaTeX installation if you
have one (it falls back to mathtext automatically if not). The UI uses Tkinter, which ships with standard
Python on Windows and macOS; on Linux install the system `python3-tk` package.
All actions also have keyboard shortcuts and a menu bar (File / Edit / Actions /
View): **Generate** (`⌘↩` / `F5`), **Dualize** (`⌘D`), **Diagonalize** (`⌘K`),
**Load** (`⌘O`), **Save** (`⌘S`), **Quit** (`⌘Q`) — `Ctrl` instead of `⌘` on
Windows/Linux. The **Edit** menu copies the Hamiltonian (LaTeX or SymPy) and the
commutators to the clipboard; **File** saves the schematic and exports the
eigenenergies to CSV (the spectrum window also has *Save plot…*). Parse errors
are shown inline with the offending netlist line highlighted (no dialog), and
the last netlist, parameters and window size are remembered between sessions. To produce a double-clickable app, freeze it with
[PyInstaller](https://pyinstaller.org) using the bundled spec (run from the repo
root): `pip install pyinstaller` then `pyinstaller fluxcharge-gui.spec --noconfirm`,
which writes `dist/fluxcharge-gui.app` (macOS) or `dist/fluxcharge-gui/`
(Windows/Linux). Running it as a bundled app — rather than the script from a
terminal — makes it a proper foreground GUI app, so mouse clicks are delivered
reliably on macOS. Set `FLUXCHARGE_DEBUG=1` to log GUI actions to
`~/fluxcharge_gui.log` for troubleshooting.

The **Dualize** button replaces the circuit with its LCG dual and regenerates;
press it again to return.

The desktop app also exposes the analysis features: a **Circuits** menu loads a
ready-made circuit from the library; the numerics panel has a **physical units**
toggle (enter `70fF` / `150nH` / `15GHz` and read the spectrum in GHz) and a
**sweep** row (plot the spectrum versus any parameter or bias, e.g. flux or
offset charge); and **Diagonalize** reports the eigenenergies together with
charge matrix elements and the bias sensitivity `df01/d(bias)` (which is zero at
a sweet spot). External flux / offset charge are entered as netlist directives
(`flux <loop>` / `offset <node>`).

---

## Circuit duality

The package implements the manuscript's **LCG duality transform** (Sec. "Circuit
Duality"): it exchanges flux and charge, so `A -> Bᵀ`, `B -> Aᵀ` (vertices become
faces and faces become vertices), `C <-> L` and `JJ <-> QPS` with each scalar
value preserved, and every gyration ratio `G -> -1/G`. The result is a new
circuit whose Hamiltonian is unitarily related to the original (so it has the
same spectrum), and the map is an involution (up to a global edge-orientation
reversal).

Dualization needs the full planar embedding (every edge bordering two faces). If
a netlist declares only the inner faces -- as the transmon does, with its single
loop -- the implicit outer face is **completed automatically**, so any circuit
that reduces to a Hamiltonian can also be dualized.

```python
from fluxcharge import from_netlist, dual, to_netlist
ckt = from_netlist("circulator.txt")
d = dual(ckt)                       # a new Circuit
print(to_netlist(d))                # its netlist
d.hamiltonian()                     # reduces like any other circuit
```

It is also available from the command line and the GUI:

```bash
fluxcharge circulator.txt --dual    # analyze the dual instead
```

Because `B* A* = (B A)ᵀ = 0`, the dual automatically satisfies Kirchhoff
exactness. Per the manuscript's unit convention (`G₀ = 1`), the scalar value of
each reciprocal element is preserved across the map (`C* = L`, `L* = C`), so a
dual inductor carries the original capacitance value as its inductance.

---

## Quick start

```python
import sympy as sp
from fluxcharge import Circuit

ckt = Circuit()

# add_<element>(edge_name, tail_vertex, head_vertex, value)
ckt.add_josephson("e1", "v1", "v2", EJ="E_J")
ckt.add_capacitor("e2", "v2", "v3", C="C")
ckt.add_capacitor("e3", "v3", "v1", C="C")

# a gyrator couples an *ordered pair* of half-edges (e1, e2)
ckt.add_gyrator(("e4", "v1", "v3"), ("e5", "v2", "v3"), G="G")

# declare the faces of the planar embedding as signed edge lists:
# "+e" if edge and loop agree in orientation, "-e" if opposed
ckt.add_loop("f1", ["+e3", "+e4"])
ckt.add_loop("f2", ["+e1", "-e4", "+e5"])
ckt.add_loop("f3", ["+e2", "-e5"])
ckt.add_loop("f4", ["-e1", "-e2", "-e3"])

# reduce to the Hamiltonian.  Grounding a node (global-flux gauge) and opening a
# loop (global-charge gauge) are optional gauge choices: one node flux and one
# loop charge are always redundant, and the reduction finds and removes those
# redundancies (as linear combinations of the remaining coordinates) on its own,
# along with the null-vector and Noether constraints and the cyclic coordinates.
# Naming a ground/open only fixes *which* representative is used; it never
# changes whether the reduction completes.
result = ckt.hamiltonian(ground="v1", open_loops="f4")   # or just ckt.hamiltonian()

print(result.H)
# -E_J*cos(phi_v2) + G**2*phi_v2**2/(2*C) + G*phi_v2*q_f3/C + q_f3**2/C
print(result.conjugate_pairs)
# [(phi_v2, q_f3, -1)]      # (coordinate, partner, symplectic coefficient)
print(result.complete)
# True                      # reduced symplectic form is non-degenerate
print(result.commutators())
# [(phi_v2, q_f3, I*hbar)]  # canonical commutation relations
```

`result.commutators()` returns the canonical commutation relations implied by
the reduced symplectic form — `[x, p] = i*hbar` (the manuscript's convention),
read off as `[a, b] = -i*hbar / coefficient` for each pair, so they are correct
even before canonicalization. `result.compact_coordinates()` lists any
coordinate sitting inside a cosine: such a variable may live on `S¹`, in which
case the relation for it becomes the exponential form
`[e^{i x / n}, p] = -(hbar/n) e^{i x / n}` and the periodicity is a physical
modelling choice (manuscript, Sec. "Constraints and Quantization").

`result.report()` prints every step that was taken — the gauge choices, the
constraints with their type (`null-vector` or `noether`), the cyclic
coordinates dropped, the coordinates eliminated, and the commutation relations:

```python
print(result.report())
```

**Canonical form.** A reduction can leave a conjugate pair with a non-unit
symplectic coefficient (for example parallel capacitors give a surviving charge
scaled by `(C1+C2)/C2`), so reading frequencies straight off `result.H` can
mislead. `result.canonical()` (or `hamiltonian(..., canonical=True)`) rescales
each pair to unit coefficient and returns a directly usable Hamiltonian;
`result.is_canonical` reports whether rescaling was needed.

**Getting numbers out.** `result.H` is a SymPy expression. Substitute values
with `result.H.subs({sp.Symbol("C"): 1.0, ...})`, turn it into a fast numeric
function with `sympy.lambdify`, or use the built-in **numerical diagonalization
and plotting** (next section). You can also hand the canonical `H` and its
conjugate pairs to an external solver such as scqubits or QuTiP.

Two runnable scripts are in [`examples/`](examples):
`circulator_jj.py` (the manuscript example, with all intermediate matrices
printed) and `lc_oscillator.py` (the linear sanity check).

---

## Numerical diagonalization and plotting

Once a circuit has reduced to a complete Hamiltonian you can diagonalize it
numerically and plot its spectrum, directly from the `ReductionResult`. This
needs the optional `numpy` dependency (`pip install "fluxcharge[numeric]"`);
plotting uses the always-installed matplotlib.

```python
import numpy as np
from fluxcharge import Circuit

ckt = Circuit()
ckt.add_josephson("e1", "v1", "v2", EJ="E_J")
ckt.add_capacitor("e2", "v1", "v2", C="C")
ckt.add_loop("f1", ["+e1", "-e2"])
res = ckt.hamiltonian(ground="v1")

res.modes()                                  # mode-type detection (see below)
res.eigenenergies({"E_J": 15.0, "C": 1.0}, n_levels=6)
res.plot_potential_wavefunctions({"E_J": 15.0, "C": 1.0})   # -> matplotlib Axes
res.plot_spectrum("q_f1", np.linspace(-1, 1, 41),           # charge dispersion
                  {"E_J": 1.0, "C": 1.0}, relative=True)
```

The free functions `eigenenergies`, `eigensystem`, `sweep`,
`hamiltonian_matrix`, `plot_energy_levels`, `plot_spectrum` and
`plot_potential_wavefunctions` are also importable from the top level. Pass
`cutoffs={symbol: size}` to set per-mode basis truncations and `offsets={charge:
n_g}` for an offset charge. A runnable demo is in
[`examples/numerical_spectrum.py`](examples/numerical_spectrum.py), and the
desktop app has a **Diagonalize** panel.

### Mode-type detection

The basis that *correctly* diagonalizes a degree of freedom is fixed by how its
flux enters the potential. `classify_modes` (or `result.modes()`) splits each
conjugate pair into one of, symmetrically in flux and charge:

| structure of the pair in `H`         | mode type       | basis                     |
|--------------------------------------|-----------------|---------------------------|
| `phi**2` and `q**2` terms            | `EXTENDED`      | harmonic oscillator       |
| `q**2`, `cos(phi)`, no `phi**2`      | `PERIODIC`      | charge basis (transmon)   |
| `phi**2`, `cos(q)`, no `q**2`        | `DUAL_PERIODIC` | flux basis (QPS + L)      |
| flux/charge in no potential term     | `FREE`          | charge basis, offset only |

This is the taxonomy used by scqubits, specialized to the flux-charge symmetric
setting: because either variable can sit inside a cosine, a quantum phase slip
gives a `DUAL_PERIODIC` (flux-basis) mode that has no analogue there. Override
the automatic choice with `mode_types={flux_symbol: "periodic"}` etc.

The solver is validated against analytics and against itself: the LC oscillator
reproduces `omega*(n+1/2)` exactly; a transmon (charge basis) and its LCG dual,
a quantum phase slip shunting an inductor (flux basis), give **identical
spectra**; and a gyrator + LC reproduces the exact quadratic frequency
`sqrt(a*c - b**2)`, validating the bilinear gyrator cross term.

**Assumptions, surfaced not hidden.** Basis cutoffs are truncations (converge
them). A gyrator's bilinear cross term `G*phi*q` is taken in the Hermitian
Weyl-symmetrized form `(phi*q + q*phi)/2`, which is the unique correct ordering
for the quadratic-plus-cosine Hamiltonians this package produces; a genuinely
ambiguous higher-degree monomial would warn. `S^1` compactness of a periodic
mode follows from the flux appearing only inside a cosine and is a physical
modelling choice (overridable via `mode_types=`).

---

## External flux and offset charge

External biases are the manuscript's **nonzero Noether constants**: an external
flux threading a **loop**, and its LCG dual, an offset (gate) charge on a
**node**.

```python
ckt.set_flux_bias("f1")            # external flux through loop f1 (symbol phi_ext_f1)
ckt.set_flux_bias("f1", "phi_ext") # ... or name/number it
ckt.set_offset_charge("v2")        # offset charge on node v2 (symbol n_g_v2)
```

or in a netlist, `flux <loop> [value]` and `offset <node> [value]`. They are
injected as constant offsets to the edge fluxes/charges (split evenly over the
loop's inductive / node's capacitive edges) so that **the symbol equals the
physical bias**: the external flux has period `2*pi` with a fluxonium sweet spot
at `pi`, and the offset charge has period `1` in Cooper-pair number. The bias
symbols flow as ordinary parameters into the symbolic `H` and the numerics, so
you sweep them like any other parameter:

```python
phi_ext = ckt.set_flux_bias("f1")
res = ckt.hamiltonian(ground="v1", open_loops="f3")     # fluxonium
res.plot_spectrum(phi_ext, np.linspace(0, 2*np.pi, 81),  # flux-tuning spectrum
                  {"E_J": 4.0, "L": 1.0, "C": 1.0}, relative=True)
```

`dual()` carries a bias across the loop↔node swap (an external flux becomes an
offset charge on the dual node and vice versa), so the dual circuit stays a
faithful image including its biases.

---

## Circuit library and tutorial

Ready-made circuits are in `fluxcharge.library`, each fully wired (elements,
loops, gauge, bias) so you can go straight to `.hamiltonian()` and a GHz
spectrum:

```python
from fluxcharge import library
res = library.fluxonium().hamiltonian(ground="v1", open_loops="f3")
res.eigenenergies(library.fluxonium().natural_params({"E_J": "5GHz", "C": "1GHz", "L": "1GHz"}))
```

`transmon`, `cooper_pair_box`, `fluxonium`, `lc_resonator`, the manuscript
`circulator`, and a quantum `phase_slip_qubit` (the LCG dual of the transmon)
are included (`library.CIRCUITS` lists them). A guided
[`examples/tutorial.ipynb`](examples/tutorial.ipynb) walks from a symbolic
Hamiltonian to spectra, flux/charge sweeps, wavefunctions, the gyrator
circulator, the phase-slip duality, and matrix elements / T1; the same content
runs as [`examples/qubit_library.py`](examples/qubit_library.py).

### Importing scqubits circuits

`from_scqubits_yaml` reads scqubits' branch YAML (`C` / `L` / `JJ` branches,
node `0` = ground) into a fluxcharge circuit, so existing scqubits circuits drop
in (loops are auto-inferred, nothing else needed):

```python
from fluxcharge import from_scqubits_yaml
ckt, params = from_scqubits_yaml("branches:\n- [JJ, 1, 0, 15]\n- [C, 1, 0, 0.3]\n")
ckt.hamiltonian(ground="0").eigenenergies(params)        # GHz, matches scqubits
# ... then add what scqubits cannot represent:
ckt.add_qps("qps", "1", "0", ES="E_S")                   # a quantum phase slip
```

Branch energies use the textbook convention (`E_C = e²/2C`, `E_L = (Φ₀/2π)²/L`),
so imports match scqubits' predefined `Transmon`/`Fluxonium` classes to
machine precision (see [`examples/compare_scqubits.py`](examples/compare_scqubits.py)).
Values may be given as energies or with scqubits' unit suffixes (`EC = 90 fF`,
`EL = 5 nH`, `EJ = 15 GHz`), which are converted to the right energy.
Only reciprocal `C`/`L`/`JJ` branches exist in scqubits; once imported you can
add gyrators and quantum phase slips, which it has no element for.

### Matrix elements and coherence

`result.matrix_elements("q_f1", params)` gives exact `<i|n|j>`;
`result.transition_sensitivity(bias, params)` gives `df01/d(bias)` (zero at a
sweet spot); `result.t1(params, noise_op, S)` is Fermi's golden rule for a
supplied spectral density `S(omega)`; and `result.dephasing_1_over_f(bias,
params, amplitude)` estimates `1/f` dephasing. The exact pieces (matrix
elements, sensitivity) are convention-free; the rate estimates are only as good
as the noise model you supply.

---

## Drawing a circuit

Two views are available.

**Lumped-element schematic (recommended).** With

```
pip install "fluxcharge[schematic]"     # schemdraw + networkx + matplotlib
```

`schematic()` draws the circuit with real symbols (inductor coil, capacitor
plates, the Josephson boxed-X, a charge-dual symbol for the quantum phase slip,
and gyrator half-edges) and straight wires, via
[schemdraw](https://schemdraw.readthedocs.io). Because circuits are planar, the
layout is built from the face structure in `B`: the outer face is placed on a
convex polygon and the interior nodes by the barycentric (Tutte) method, giving
a crossing-free drawing, and edges on the outer face are routed around the
outside so their interior parallel siblings stay straight.

```python
ckt.schematic(path="circuit.png")
# the outer face is auto-detected (largest loop); name it explicitly if needed,
# and/or supply node coordinates for full control:
ckt.schematic(path="circuit.png", outer_loop="f4")
ckt.schematic(path="circuit.png", positions={"v1": (8, 0), "v2": (8, 6), "v3": (0, 0)})
```

An LC loop draws as a rectangle, parallel branches as a ladder, and the
manuscript circulator as the Josephson junction across the top with the
capacitors down the sides and the gyrator half-edges as interior diagonals. For
the cleanest result on a dense or irregular circuit, pass explicit `positions=`
-- which is also the natural hook for an interactive (drag-and-drop) editor: the
editor owns the coordinates, `schematic()` draws the symbols and straight wires.
See [`examples/schematic.py`](examples/schematic.py).

**Topology graph.** `to_networkx()` returns a `networkx.MultiDiGraph`
reconstructed from the incidence matrix `A` (nodes, oriented edges) with the
faces read from `B` on `G.graph["loops"]`; `draw()` renders that graph. This is
the data model to build the editor on, and is described in
[`examples/draw_circuit.py`](examples/draw_circuit.py). Install its deps with
`pip install "fluxcharge[viz]"`.

---

## Inputting a circuit

A circuit is an oriented planar multigraph `G = (V, E, F)`.

* **Vertices** `V` are created implicitly the first time you name them.
* **Edges** `E` are added through the element methods. Each edge is directed
  `tail → head`; this fixes the sign convention of the incidence matrix
  (`A[e, head] = +1`, `A[e, tail] = −1`).
* **Loops / faces** `F` may be declared with `add_loop(name, signed_edges)` --
  the signs are exactly the entries `B[l, e]` of the orientation matrix (`"+e"`
  when the edge and loop circulate the same way, `"-e"` otherwise) -- but this is
  **optional**. If you declare no loops, they are inferred from the graph:
  `hamiltonian()` calls `infer_loops()`, which finds a planar embedding and
  traces its faces (so `dual()` and `schematic()` work). A non-planar circuit
  falls back to a cycle basis (the Hamiltonian still works; duality/drawing do
  not) and warns. So you can build a circuit from just its elements:

  ```python
  ckt = Circuit()
  ckt.add_josephson("e1", "v1", "v2", EJ="E_J")
  ckt.add_capacitor("e2", "v1", "v2", C="C")
  ckt.hamiltonian(ground="v1")        # loops inferred automatically
  ```

| element              | method            | class       | energy              |
|----------------------|-------------------|-------------|---------------------|
| capacitor            | `add_capacitor`   | capacitive  | `Q²/2C`             |
| inductor             | `add_inductor`    | inductive   | `Φ²/2L`             |
| Josephson junction   | `add_josephson`   | inductive   | `−E_J cos Φ`        |
| quantum phase slip   | `add_qps`         | capacitive  | `−E_S cos Q`        |
| gyrator (edge pair)  | `add_gyrator`     | gyrative    | none (couples Φ, Q) |

Values can be a SymPy symbol, a number, or a string (parsed by SymPy). Omit a
value to get a fresh positive symbol.

### Conventions

Following the manuscript, the reference conductance is set to `G₀ = 1` and the
reduced flux quantum to `1`, so fluxes and charges are dimensionless and
interchangeable. A Josephson junction is an inductive (non-linear) edge; a
coherent quantum phase slip is its capacitive dual.

---

## What is inspectable

Every intermediate object is available as a SymPy matrix or expression:

```python
ckt.incidence_matrix()      # A   (|E| x |V|)
ckt.orientation_matrix()    # B   (|F| x |E|),  with  B @ A == 0
ckt.connection_matrix()     # M = (1/2) B (P_C - P_I) A
ckt.omega()                 # the antisymmetric form Omega
ckt.edge_flux("e1")         # Phi_e  = (A phi)_e
ckt.edge_charge("e1")       # Q_e    = (B^T q)_e
ckt.energy()                # total energy E(phi, q)
ckt.lagrangian()            # the first-order Lagrangian
```

For full manual control over the reduction, drive the `Reducer` yourself:

```python
from fluxcharge import Reducer

r = Reducer(ckt)
r.ground("v1")                       # set phi_v1 = 0
r.open_loop("f4")                    # set q_f4 = 0
r.auto_constraints()                 # derive + impose null-vector and Noether
                                     # constraints, drop cyclic coordinates
# or impose your own constraint, with the velocity handled automatically:
# r.impose("q_f1", "G*phi_v2 + q_f3")
result = r.to_hamiltonian()
```

You can inspect the building blocks directly: `r.energy()`,
`r.grad_energy()`, `r.momentum_coefficients()`, `r.symplectic_matrix()`,
`r.symmetry_directions()`, and `r.derive_constraints()` (which returns each
constraint tagged `"null-vector"` or `"noether"`).

---

## Scope and honesty about the reduction

The reduction implements the constraint analysis of Section "Constraints and
Quantization" of the manuscript. Starting from the first-order Lagrangian
`L = ½⟨x|Ω|ẋ⟩ − E(x)`, the principle of least action along a direction
`X` reads `½⟨X|Ω|ẋ⟩ = ⟨X|∇E⟩` (Eq. plac), which yields two constraint
families that the package finds and classifies:

* **Null-vector constraints** — `X` a null vector of `Ω`, so the left side
  vanishes and the constraint is the purely algebraic `⟨X|∇E⟩ = 0`. These are
  the *pure* (single element class) and *gyrator-mixing* null vectors. The
  pure-capacitor-loop / pure-inductor-cut cases that make the method of nodes
  or loops singular are handled here, and are covered by the test suite.

* **Noether constraints** — `X` a symmetry of the energy (it leaves every
  capacitive edge charge and inductive edge flux invariant) but not a null
  vector of `Ω`, so the right side vanishes and the velocity relation
  `⟨X|Ω|ẋ⟩ = 0` integrates to a conservation law.

It is useful to be clear about which parts are automatic and exact and which
require your judgement:

* **Fully automatic and exact.** The boundary maps `A`, `B`, the connection
  matrix `M`, the form `Ω`, the Lagrangian, the energy, the symplectic matrix
  `f = −Ω/2`, the null vectors and energy symmetries, the resulting
  null-vector and Noether constraints, the elimination of cyclic coordinates,
  and the final non-degeneracy (completeness) check. These are validated
  against the manuscript's worked example and a pure-null-vector circuit.

* **Your (optional) physical input.** The **ground node** (a global-flux gauge)
  and which **loop(s) to open** (a global-charge gauge) are gauge choices you may
  set, but you don't have to: one node flux and one loop charge are always
  redundant, and the reduction detects and removes those redundancies itself —
  as linear combinations of the remaining coordinates when the leftover isn't a
  single coordinate — so the reduction completes for any choice or none. Naming
  them only selects which representative survives. The schematic's outer face is
  taken from the planar embedding and is independent of any gauge choice.

* **A reported choice.** When a constraint could be solved for more than one
  coordinate, the reducer picks one deterministically (charges before fluxes,
  earliest-declared first), **reports it** in `result.report()`, and lets you
  override it with `keep=[...]` or `eliminate=[...]`. Different valid choices
  give physically equivalent Hamiltonians in different coordinates.

* **The completeness guarantee.** After reduction the package checks that the
  reduced symplectic form is non-degenerate. If constraints remain unresolved
  (for example a gauge still needs choosing) `hamiltonian()` raises by default
  rather than return a wrong Hamiltonian; pass `strict=False` to inspect the
  partial result. A coordinate may survive with a non-unit symplectic
  coefficient (reported per conjugate pair); rescaling that coordinate puts the
  pair in canonical form — e.g. parallel capacitors combine to `C₁+C₂` once the
  surviving charge is canonically normalized.

This package automates the mechanical, provably-correct core and gives you
inspectable, scriptable control over the genuinely physical steps. Fully
automatic reduction of *arbitrary* strongly non-linear circuits is a research
problem; for such cases, use the `Reducer` moves to guide the reduction and
check `result.report()`. The Hamiltonian is always returned as an explicit
SymPy expression you can verify by hand.

---

## Tests

```bash
pip install pytest
pytest tests/
# or, with no pytest:
python tests/test_example.py
```

The suite checks Kirchhoff exactness (`B A = 0`), antisymmetry of `Ω`, the
Lagrangian against the manuscript term-by-term, the reduced Hamiltonian against
the published circulator result, a pure-null-vector circuit, and a set of
textbook circuits (LC oscillator, transmon, fluxonium, and a quantum-phase-slip
circuit) against their known Hamiltonians. `examples/reproduce_manuscript.py`
regenerates the manuscript's worked example end to end and asserts an exact
match.

---

## How to cite

If you use this package, please cite both the software and the accompanying
paper. A [`CITATION.cff`](CITATION.cff) is included (GitHub renders a "Cite this
repository" button from it); fill in the Zenodo and paper DOIs once the release
is archived and the paper is posted. The accompanying paper is *"Gyrators for
superconducting circuit design"* (Salcedo, Cocquyt, Osborne, Houck).

---

## Layout

```
fluxcharge/
  __init__.py        package surface: Circuit, elements, Reducer, drawing
  elements.py        Capacitor, Inductor, JosephsonJunction,
                     QuantumPhaseSlip, Gyrator
  circuit.py         Circuit: boundary maps, M, Omega, Lagrangian,
                     .validate(), .hamiltonian(), .schematic(), .to_networkx()
  reduction.py       Reducer + ReductionResult (null-vector / Noether reduction,
                     completeness check, .canonical())
  visualize.py       networkx topology graph (data model for an editor)
  schematic.py       planar lumped-element schematic (schemdraw)
  netlist.py         text netlist parser (from_netlist / parse_netlist)
  __main__.py        CLI: netlist -> drawing + Lagrangian + Hamiltonian
examples/
  circulator.txt           a netlist for the CLI workflow
  reproduce_manuscript.py  rebuilds the paper's worked example, asserts a match
  circulator_jj.py   the manuscript's non-reciprocal example
  lc_oscillator.py   linear sanity check
tests/
  test_example.py    validation against the manuscript
```

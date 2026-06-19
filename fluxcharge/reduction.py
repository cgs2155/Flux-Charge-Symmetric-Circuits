"""
Reduction of the flux-charge symmetric Lagrangian to a Hamiltonian.

This module implements the constraint reduction of Section "Constraints and
Quantization" of

    "Gyrators for superconducting circuit design",
    C. Salcedo, S. Cocquyt, A. Osborne, A. A. Houck.

The Lagrangian built by :class:`~fluxcharge.circuit.Circuit` is first order in
the velocities,

    L(x, xdot) = (1/2) <x| Omega |xdot>  -  E(x),

with ``x`` the node fluxes and loop charges and ``Omega`` the antisymmetric
form of Eq. (omega).  ``Omega`` is degenerate, and the reduction proceeds by
the principle of least action applied along a direction
``X = (c_v ; d_l)`` (Eq. plac):

    (1/2) <X| Omega |xdot>  =  <X| grad E>.

This single equation gives the paper's two constraint families:

* **Null-vector constraints.**  If ``X`` is a null vector of ``Omega`` the
  left-hand side vanishes identically and the equation becomes the purely
  algebraic constraint ``<X| grad E> = 0``.  There is one for every null
  vector of ``Omega`` carrying a non-trivial energy gradient; these are the
  *pure* (single element class, from loops/cuts) and *gyrator-mixing* null
  vectors of the paper's taxonomy.

* **Noether constraints.**  If instead ``X`` is a symmetry of the energy
  (``<X| grad E>`` vanishes identically -- i.e. ``X`` leaves every capacitive
  edge charge and inductive edge flux invariant) and ``X`` is *not* a null
  vector of ``Omega``, the equation becomes ``<X| Omega |xdot> = 0``, a linear
  velocity relation that integrates to the conservation law
  ``<X| Omega |x> = const``.  The constant is gauge-fixed to zero.

Two further ingredients complete the reduction:

* **Gauge directions** -- the global flux shift (a ground node) and global
  charge shift (an open loop) per really-connected component.  These are
  physical choices supplied by the user via :meth:`Reducer.ground` and
  :meth:`Reducer.open_loop`.

* **Cyclic coordinates** -- coordinates absent from ``E`` are ignorable; once
  their conservation law has been used they are dropped, completing the
  symplectic reduction.

After substitution the Hamiltonian is the energy in the surviving coordinates
(the velocity-free part of the reduced Lagrangian), and the reduction is
verified by checking that the reduced symplectic form is non-degenerate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp

from .elements import CAPACITIVE, INDUCTIVE


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def velocity_free_part(expr: sp.Expr, velocities: Sequence[sp.Symbol]) -> sp.Expr:
    """Sum of the terms of *expr* containing none of the *velocities*."""
    expr = sp.expand(expr)
    vels = set(velocities)
    keep = [t for t in expr.as_ordered_terms() if not (t.free_symbols & vels)]
    return sp.Add(*keep) if keep else sp.Integer(0)


# ----------------------------------------------------------------------
# result container
# ----------------------------------------------------------------------
@dataclass
class ReductionResult:
    """The outcome of a reduction.

    Attributes
    ----------
    H : sympy expression
        The Hamiltonian in the surviving coordinates.
    coordinates : list of sympy symbols
        The surviving (reduced) coordinates.
    conjugate_pairs : list of (symbol, symbol, sympy expression)
        Canonically conjugate ``(coordinate, partner, symplectic_coefficient)``
        triples read off the reduced symplectic form.  A coefficient of
        ``+/-1`` means the pair is already canonically normalized.
    complete : bool
        True iff the reduced symplectic form is non-degenerate (even
        dimension, full rank) -- i.e. the reduction is finished and ``H`` is a
        genuine Hamiltonian on the reduced phase space.
    symplectic_matrix : sympy.Matrix
        The reduced antisymmetric form on ``coordinates``.
    constraints : list of (str, sympy.Eq)
        Each imposed constraint with its type, ``"null-vector"`` or
        ``"noether"``.
    gauge : dict
        Gauge substitutions applied (``{symbol: 0}``).
    cyclic : list of sympy symbols
        Coordinates dropped as cyclic (absent from the energy).
    eliminated : dict
        ``{coordinate: expression}`` for every coordinate solved away.
    lagrangian_reduced : sympy expression
        The Lagrangian after all substitutions.
    operations : list of str
        Human-readable log of the moves applied, in order.
    """

    H: sp.Expr
    coordinates: List[sp.Symbol]
    conjugate_pairs: List[Tuple[sp.Symbol, sp.Symbol, sp.Expr]] = field(default_factory=list)
    complete: bool = False
    symplectic_matrix: Optional[sp.Matrix] = None
    constraints: List[Tuple[str, sp.Eq]] = field(default_factory=list)
    gauge: Dict[sp.Symbol, sp.Expr] = field(default_factory=dict)
    cyclic: List[sp.Symbol] = field(default_factory=list)
    eliminated: Dict[sp.Symbol, sp.Expr] = field(default_factory=dict)
    lagrangian_reduced: Optional[sp.Expr] = None
    operations: List[str] = field(default_factory=list)

    def __repr__(self):
        flag = "" if self.complete else " [INCOMPLETE]"
        return f"ReductionResult(H={self.H}, coordinates={self.coordinates}){flag}"

    def _repr_latex_(self):  # pragma: no cover - cosmetic
        return "$H = " + sp.latex(self.H) + "$"

    @property
    def is_canonical(self) -> bool:
        """True iff every conjugate pair has symplectic coefficient +/-1."""
        return all(abs(sp.nsimplify(c)) == 1 for *_, c in self.conjugate_pairs) \
            if self.conjugate_pairs else False

    def canonical(self) -> "ReductionResult":
        """Return a copy in which every conjugate pair has unit symplectic
        coefficient.

        Reduction can leave a pair with a non-unit coefficient -- for example
        parallel capacitors give a surviving charge scaled by ``(C1+C2)/C2``.
        Reading frequencies straight off such an ``H`` would be wrong.  This
        rescales the charge of each affected pair so the pair becomes
        canonically conjugate, returning the Hamiltonian an experimenter can use
        directly.  The change of variables is recorded in ``operations``.
        """
        import dataclasses

        H = self.H
        new_pairs = []
        notes = []
        for a, b, c in self.conjugate_pairs:
            cs = sp.simplify(c)
            if cs == 1 or cs == -1:
                new_pairs.append((a, b, cs))
                continue
            # rescale the charge member (prefer q_...) so the coefficient -> 1
            if str(b).startswith("q_") or not str(a).startswith("q_"):
                target, keep = b, a
            else:
                target, keep = a, b
            # the reduced symbol equals (canonical momentum)/c, so substitute
            H = sp.expand(H.subs(target, target / cs))
            new_pairs.append((keep, target, sp.Integer(1)))
            notes.append(f"canonicalize: {target} := ({cs})*{target}")
        H = sp.simplify(H)
        return dataclasses.replace(
            self,
            H=H,
            conjugate_pairs=new_pairs,
            operations=list(self.operations) + notes,
        )

    def report(self) -> str:
        """A multi-line human-readable summary of the reduction."""
        lines = ["Reduction report", "================"]
        for op in self.operations:
            lines.append(f"  - {op}")
        if self.constraints:
            lines.append("constraints (Salcedo et al., Sec. 'Constraints and Quantization'):")
            for kind, eq in self.constraints:
                lines.append(f"  [{kind}] {eq}")
        if self.gauge:
            lines.append("gauge: " + ", ".join(f"{k} = 0" for k in self.gauge))
        if self.cyclic:
            lines.append("cyclic (dropped): " + ", ".join(map(str, self.cyclic)))
        if self.conjugate_pairs:
            ps = ", ".join(f"({a}, {b}; coeff {c})" for a, b, c in self.conjugate_pairs)
            lines.append(f"conjugate pairs: {ps}")
            lines.append("commutation relations (hbar = reduced):")
            for a, b, val in self.commutators():
                lines.append(f"  [{a}, {b}] = {val}")
            compact = self.compact_coordinates()
            if compact:
                lines.append("  note: " + ", ".join(map(str, compact))
                             + " appear inside a cosine and may live on S^1; if so the"
                             " relation for that variable x becomes [e^(i x/n), p] ="
                             " -(hbar/n) e^(i x/n) (the periodicity is a physical choice).")
        lines.append(f"coordinates: {', '.join(map(str, self.coordinates))}")
        lines.append(f"complete (non-degenerate symplectic form): {self.complete}")
        lines.append(f"H = {self.H}")
        return "\n".join(lines)

    def commutators(self, hbar=None):
        """Canonical commutation relations implied by the reduced symplectic form.

        For a conjugate pair with symplectic coefficient ``c`` the 2x2 block of
        the reduced form is ``[[0, c], [-c, 0]]``; its inverse gives the Poisson
        bracket ``{a, b} = -1/c`` and hence ``[a, b] = -i*hbar/c`` (Eq.
        ``canonical`` of the manuscript).  After :meth:`canonical` every ``c`` is
        +/-1, so each relation is ``+/- i*hbar``.  All brackets between different
        pairs, and between the two members of different pairs, vanish.

        Returns a list of ``(a, b, value)`` triples.
        """
        hbar = hbar if hbar is not None else sp.Symbol("hbar", positive=True)
        out = []
        for a, b, c in self.conjugate_pairs:
            out.append((a, b, sp.simplify(-sp.I * hbar / c)))
        return out

    # ------------------------------------------------------------------
    # numerical analysis (optional; needs numpy / matplotlib)
    # ------------------------------------------------------------------
    def modes(self, mode_types=None):
        """Classify the conjugate pairs into ``Mode`` objects (mode-type
        detection).  See :func:`fluxcharge.numerics.classify_modes`."""
        from .numerics import classify_modes
        return classify_modes(self, mode_types=mode_types)

    def hamiltonian_matrix(self, params=None, **kw):
        """Dense numeric Hamiltonian matrix; see
        :func:`fluxcharge.numerics.hamiltonian_matrix`."""
        from .numerics import hamiltonian_matrix
        return hamiltonian_matrix(self, params, **kw)

    def eigenenergies(self, params=None, n_levels=6, **kw):
        """Lowest ``n_levels`` eigenenergies; see
        :func:`fluxcharge.numerics.eigenenergies`."""
        from .numerics import eigenenergies
        return eigenenergies(self, params, n_levels, **kw)

    def eigensystem(self, params=None, n_levels=6, **kw):
        """Lowest eigenvalues and eigenvectors; see
        :func:`fluxcharge.numerics.eigensystem`."""
        from .numerics import eigensystem
        return eigensystem(self, params, n_levels, **kw)

    def sweep(self, parameter, values, params=None, n_levels=6, **kw):
        """Eigenenergies versus a swept parameter; see
        :func:`fluxcharge.numerics.sweep`."""
        from .numerics import sweep
        return sweep(self, parameter, values, params, n_levels, **kw)

    def plot_spectrum(self, parameter, values, params=None, **kw):
        """Plot eigenenergies versus a swept parameter (matplotlib)."""
        from .plotting import plot_spectrum
        return plot_spectrum(self, parameter, values, params, **kw)

    def plot_energy_levels(self, params=None, **kw):
        """Draw the eigenenergies as a level diagram (matplotlib)."""
        from .plotting import plot_energy_levels
        return plot_energy_levels(self, params, **kw)

    def plot_potential_wavefunctions(self, params=None, **kw):
        """Plot the potential and eigenstate densities (single-mode)."""
        from .plotting import plot_potential_wavefunctions
        return plot_potential_wavefunctions(self, params, **kw)

    def compact_coordinates(self):
        """Surviving coordinates that appear inside a cosine of ``H`` -- the
        candidates for a periodic (``S^1``) identification, for which the naive
        commutator must be replaced by the exponential form (manuscript,
        Sec. 'Constraints and Quantization')."""
        cos_args = [t.args[0] for t in self.H.atoms(sp.cos)]
        compact = []
        for sym in self.coordinates:
            if any(sym in arg.free_symbols for arg in cos_args):
                compact.append(sym)
        return compact


# ----------------------------------------------------------------------
# the reducer
# ----------------------------------------------------------------------
class Reducer:
    """Faithful constraint reduction of a circuit's Lagrangian.

    Usually driven through :meth:`fluxcharge.circuit.Circuit.hamiltonian`.
    For manual control the moves can be scripted::

        r = Reducer(circuit)
        r.ground("v1")             # global-flux gauge
        r.open_loop("f4")          # global-charge gauge
        r.auto_constraints()       # derive null-vector + Noether constraints
        result = r.to_hamiltonian()

    or constraints can be supplied by hand with :meth:`impose`.
    """

    def __init__(self, circuit):
        self.circuit = circuit
        phi, q, phidot, qdot = circuit.coordinate_symbols()
        self.coords: List[sp.Symbol] = list(phi) + list(q)
        self.vels: List[sp.Symbol] = list(phidot) + list(qdot)
        self.velmap: Dict[sp.Symbol, sp.Symbol] = dict(zip(self.coords, self.vels))
        self.fluxes = list(phi)
        self.charges = list(q)
        self.L = sp.expand(circuit.lagrangian())

        self._constraints: List[Tuple[sp.Symbol, sp.Expr]] = []   # ordered coord -> expr
        self._constraint_meta: List[Tuple[str, sp.Eq]] = []
        self._gauge: Dict[sp.Symbol, sp.Integer] = {}
        self._cyclic: List[sp.Symbol] = []
        self._eliminated: Dict[sp.Symbol, sp.Expr] = {}
        self._ops: List[str] = []

    # ------------------------------------------------------------------
    # symbol helpers
    # ------------------------------------------------------------------
    def _flux_symbol(self, node: str) -> sp.Symbol:
        if node in self.circuit.vertices:
            return sp.Symbol(f"phi_{node}")
        raise ValueError(f"unknown node {node!r}")

    def _charge_symbol(self, loop: str) -> sp.Symbol:
        if loop in self.circuit.loops:
            return sp.Symbol(f"q_{loop}")
        raise ValueError(f"unknown loop {loop!r}")

    def _velocity_of(self, expr: sp.Expr) -> sp.Expr:
        """Time derivative of *expr* via the chain rule in the coordinates."""
        expr = sp.expand(expr)
        out = sp.Integer(0)
        for c in self.coords:
            d = sp.diff(expr, c)
            if d != 0:
                out += d * self.velmap[c]
        return sp.expand(out)

    # ------------------------------------------------------------------
    # exact building blocks
    # ------------------------------------------------------------------
    def energy(self) -> sp.Expr:
        """The circuit energy ``E(x) = -(velocity-free part of L)``."""
        return -velocity_free_part(self.L, self.vels)

    def grad_energy(self) -> sp.Matrix:
        E = self.energy()
        return sp.Matrix([sp.diff(E, c) for c in self.coords])

    def momentum_coefficients(self) -> List[sp.Expr]:
        """``a_i = dL/d(xdot_i)``."""
        return [sp.diff(self.L, v) for v in self.vels]

    def symplectic_matrix(self) -> sp.Matrix:
        """Presymplectic matrix ``f_{ki} = d a_k/d x_i - d a_i/d x_k`` (= -Omega/2)."""
        a = self.momentum_coefficients()
        n = len(self.coords)
        f = sp.zeros(n, n)
        for k in range(n):
            for i in range(n):
                f[k, i] = sp.diff(a[k], self.coords[i]) - sp.diff(a[i], self.coords[k])
        return f

    def symmetry_directions(self) -> List[sp.Matrix]:
        """Directions ``X = (c; d)`` leaving the energy invariant.

        ``E`` is a sum of functions of capacitive edge charges and inductive
        edge fluxes, so it is invariant exactly when every such edge variable
        is: ``(A c)_e = 0`` for inductive ``e`` and ``(B^T d)_e = 0`` for
        capacitive ``e``.  The null space of those rows is the symmetry space.
        """
        A = self.circuit.incidence_matrix()
        B = self.circuit.orientation_matrix()
        nV, nF = len(self.circuit.vertices), len(self.circuit.loops)
        eidx = {e: i for i, e in enumerate(self.circuit.edges)}
        rows = []
        for ename in self.circuit.edges:
            cls = self.circuit._edges[ename].edge_class
            ei = eidx[ename]
            if cls == INDUCTIVE:
                rows.append([A[ei, v] for v in range(nV)] + [0] * nF)
            elif cls == CAPACITIVE:
                rows.append([0] * nV + [B[l, ei] for l in range(nF)])
        if not rows:
            return sp.eye(nV + nF).columnspace()
        return sp.Matrix(rows).nullspace()

    def derive_constraints(self) -> List[Tuple[str, sp.Expr]]:
        """The null-vector and Noether constraint forms (each equals zero).

        Returns a list of ``(type, expr)`` with ``type`` in
        ``{"null-vector", "noether"}``.
        """
        Omega = self.circuit.omega()
        gE = self.grad_energy()
        x = sp.Matrix(self.coords)
        out: List[Tuple[str, sp.Expr]] = []

        # null-vector constraints: X in ker(Omega), <X|grad E> not identically 0
        for X in Omega.nullspace():
            g = sp.expand((X.T * gE)[0])
            if sp.simplify(g) != 0:
                out.append(("null-vector", g))

        # Noether constraints: X a symmetry of E, not a null vector of Omega
        for X in self.symmetry_directions():
            if sp.simplify((X.T * gE)[0]) != 0:
                continue
            rel = sp.expand((X.T * Omega * x)[0])
            if sp.simplify(rel) != 0:
                out.append(("noether", rel))
        return out

    # ------------------------------------------------------------------
    # scriptable moves
    # ------------------------------------------------------------------
    def ground(self, *nodes: str) -> "Reducer":
        """Set node flux(es) to zero (global-flux gauge)."""
        for node in nodes:
            self._gauge[self._flux_symbol(node)] = sp.Integer(0)
            self._ops.append(f"ground({node})")
        return self

    def open_loop(self, *loops: str) -> "Reducer":
        """Set loop charge(s) to zero (global-charge gauge)."""
        for loop in loops:
            self._gauge[self._charge_symbol(loop)] = sp.Integer(0)
            self._ops.append(f"open_loop({loop})")
        return self

    def impose(self, coord, expr, kind: str = "manual") -> "Reducer":
        """Eliminate *coord* via *expr* (its time derivative is handled)."""
        coord = sp.sympify(coord)
        expr = sp.sympify(expr)
        self._constraints.append((coord, expr))
        self._constraint_meta.append((kind, sp.Eq(coord, expr)))
        self._eliminated[coord] = expr
        self._ops.append(f"impose [{kind}] {coord} = {expr}")
        return self

    def auto_constraints(self, keep: Optional[Sequence] = None,
                         eliminate: Optional[Sequence] = None) -> "Reducer":
        """Derive and impose the null-vector and Noether constraints.

        Gauge choices already registered are applied to the constraint forms
        first.  Each independent constraint is solved for one coordinate; the
        target is chosen deterministically (charges before fluxes, earliest
        declared first) unless restricted by *keep* or steered by *eliminate*.
        """
        keep = {sp.sympify(k) for k in (keep or [])}
        prefer = [sp.sympify(e) for e in (eliminate or [])]

        derived = self.derive_constraints()

        # apply gauge, then reduce to an independent set (in the coordinates)
        kept_exprs: List[sp.Expr] = []
        kept_types: Dict[sp.Expr, str] = {}
        for kind, r in derived:
            rg = sp.expand(r.subs(self._gauge))
            if rg == 0:
                continue
            if kept_exprs:
                M = sp.Matrix([[sp.diff(e, c) for c in self.coords] for e in kept_exprs])
                v = sp.Matrix([[sp.diff(rg, c) for c in self.coords]])
                if M.rank() == M.col_join(v).rank():
                    continue
            kept_exprs.append(rg)
            kept_types[rg] = kind

        for r in kept_exprs:
            cands = [c for c in self.coords
                     if c in r.free_symbols and c not in self._gauge
                     and c not in self._eliminated and c not in keep]
            if not cands:
                continue
            def rank(s):
                pref = prefer.index(s) if s in prefer else len(prefer)
                is_flux = 1 if s in self.fluxes else 0
                return (pref, is_flux, self.coords.index(s))
            target = sorted(cands, key=rank)[0]
            sol = sp.solve(sp.Eq(r, 0), target, dict=True)
            if sol:
                self.impose(target, sol[0][target], kind=kept_types[r])

        # cyclic coordinates: absent from the energy, not gauged / eliminated
        E = self.energy()
        self._cyclic = [c for c in self.coords
                        if c not in E.free_symbols
                        and c not in self._gauge and c not in self._eliminated]
        if self._cyclic:
            self._ops.append("drop cyclic: " + ", ".join(map(str, self._cyclic)))

        # remove any residual redundancy that survives as a *linear combination*
        self._complete_reduction()
        return self

    def _complete_reduction(self, max_iter: Optional[int] = None) -> "Reducer":
        """Eliminate residual redundant directions left after gauge fixing.

        The dropped-cyclic step only catches coordinates that are *individually*
        absent from the energy.  A redundancy can instead survive as a linear
        *combination* of the surviving coordinates -- this is the generic case of
        the manuscript's global-charge (and global-flux) gauge: the faces of a
        planar circuit are linearly dependent, so one loop charge is always a
        combination of the others, and depending on which representative is fixed
        the leftover can be e.g. ``q_a + q_b`` rather than a single ``q``.

        Here we look directly at the reduced presymplectic form.  Any null
        direction ``n`` of that form that is also a symmetry of the reduced
        energy (the energy does not depend on the combination ``sum n_i x_i``) is
        a genuine gauge/redundant direction carrying no dynamics; we fix it by
        eliminating one participating coordinate via that linear combination.
        A null direction that *does* appear in the energy is a real singularity
        and is left in place, so a genuinely ill-posed circuit still reports
        ``complete = False`` rather than being silently truncated.
        """
        for _ in range(max_iter if max_iter is not None else len(self.coords) + 1):
            L_red = self._reduced_lagrangian()
            removed = set(self._gauge) | set(self._eliminated) | set(self._cyclic)
            survivors = [c for c in self.coords if c not in removed]
            if not survivors:
                return self
            f = self._reduced_symplectic(L_red, survivors)
            if len(survivors) % 2 == 0 and f.rank() == len(survivors):
                return self  # non-degenerate: reduction is complete
            null = f.nullspace()
            if not null:
                return self
            H_red = sp.expand(-velocity_free_part(L_red, self.vels))
            gradH = [sp.diff(H_red, c) for c in survivors]
            acted = False
            for nvec in null:
                dderiv = sp.expand(sum(nvec[i] * gradH[i] for i in range(len(survivors))))
                if sp.simplify(dderiv) != 0:
                    continue  # this null direction carries energy: a real singularity
                nz = [i for i in range(len(survivors)) if sp.simplify(nvec[i]) != 0]
                if not nz:
                    continue
                # eliminate a participating coordinate (prefer a charge, latest declared)
                nz.sort(key=lambda i: (0 if survivors[i] in self.charges else 1,
                                       -self.coords.index(survivors[i])))
                k = nz[0]
                target = survivors[k]
                expr = -sum(nvec[i] * survivors[i]
                            for i in range(len(survivors)) if i != k) / nvec[k]
                self.impose(target, sp.simplify(sp.expand(expr)), kind="redundant")
                acted = True
                break
            if not acted:
                return self
        return self

    # ------------------------------------------------------------------
    # assembling the Hamiltonian
    # ------------------------------------------------------------------
    def _reduced_lagrangian(self) -> sp.Expr:
        L = self.L
        for coord, expr in self._constraints:
            sub = {coord: expr, self.velmap[coord]: self._velocity_of(expr)}
            L = sp.expand(L.subs(sub))
        drop = dict(self._gauge)
        drop.update({self.velmap[c]: 0 for c in self._gauge})
        for c in self._cyclic:
            drop[c] = 0
            drop[self.velmap[c]] = 0
        return sp.expand(L.subs(drop))

    def _reduced_symplectic(self, L_red: sp.Expr,
                            survivors: Sequence[sp.Symbol]) -> sp.Matrix:
        a = [sp.diff(L_red, self.velmap[c]) for c in survivors]
        n = len(survivors)
        f = sp.zeros(n, n)
        for i in range(n):
            for j in range(n):
                f[i, j] = sp.expand(sp.diff(a[i], survivors[j])
                                    - sp.diff(a[j], survivors[i]))
        return f

    @staticmethod
    def _pairs_from_form(f: sp.Matrix, survivors: Sequence[sp.Symbol]):
        pairs = []
        used = set()
        n = len(survivors)
        for i in range(n):
            if i in used:
                continue
            for j in range(i + 1, n):
                if j in used:
                    continue
                if sp.simplify(f[i, j]) != 0:
                    pairs.append((survivors[i], survivors[j], sp.simplify(f[i, j])))
                    used.add(i)
                    used.add(j)
                    break
        return pairs

    def to_hamiltonian(self) -> ReductionResult:
        """Carry out the reduction and return a :class:`ReductionResult`."""
        L_red = self._reduced_lagrangian()
        H = sp.trigsimp(sp.expand(-velocity_free_part(L_red, self.vels)))

        removed = set(self._gauge) | set(self._eliminated) | set(self._cyclic)
        survivors = [c for c in self.coords if c not in removed]

        f = self._reduced_symplectic(L_red, survivors)
        complete = (len(survivors) > 0 and len(survivors) % 2 == 0
                    and f.rank() == len(survivors))
        pairs = self._pairs_from_form(f, survivors)

        return ReductionResult(
            H=H,
            coordinates=survivors,
            conjugate_pairs=pairs,
            complete=complete,
            symplectic_matrix=f,
            constraints=list(self._constraint_meta),
            gauge=dict(self._gauge),
            cyclic=list(self._cyclic),
            eliminated=dict(self._eliminated),
            lagrangian_reduced=L_red,
            operations=list(self._ops),
        )

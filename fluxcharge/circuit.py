"""
The :class:`Circuit` object: assembles the boundary maps, the connection
matrix, the antisymmetric form Omega, and the flux-charge symmetric Lagrangian
of an LCG circuit, following

    "Gyrators for superconducting circuit design"
    C. Salcedo, S. Cocquyt, A. Osborne, A. A. Houck.

All quantities are built symbolically with sympy.

Coordinate ordering
-------------------
The configuration vector is

    x = (phi_{v_1}, ..., phi_{v_n},  q_{l_1}, ..., q_{l_m})

i.e. node fluxes first (one per vertex), then loop charges (one per face/loop).
Omega is the (n+m) x (n+m) antisymmetric matrix of Eq. (omega).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple

import sympy as sp

from .elements import (
    CAPACITIVE,
    GYRATIVE,
    INDUCTIVE,
    Capacitor,
    Edge,
    Gyrator,
    Inductor,
    JosephsonJunction,
    QuantumPhaseSlip,
)


class Circuit:
    """A lumped-element LCG circuit.

    Build a circuit by adding elements and then declaring its loops (the faces
    of the planar embedding) as signed edge lists.  The signed edge list for a
    loop records, for each edge on its boundary, ``+name`` if the edge and loop
    are oriented alike and ``-name`` if they are oriented oppositely -- exactly
    the entries B[l, e] of the orientation matrix.
    """

    def __init__(self):
        self._elements: List = []
        self._edges: "OrderedDict[str, Edge]" = OrderedDict()
        self._vertices: "OrderedDict[str, None]" = OrderedDict()
        self._loops: "OrderedDict[str, List[Tuple[int, str]]]" = OrderedDict()
        self._gyrator_pairs: List[Tuple[str, str, sp.Expr]] = []
        # external biases: constant Noether offsets (see set_flux_bias / set_offset_charge)
        self._flux_bias: "OrderedDict[str, sp.Expr]" = OrderedDict()      # loop -> Phi_ext
        self._offset_charge: "OrderedDict[str, sp.Expr]" = OrderedDict()  # node -> n_g
        self._planar = None     # set by infer_loops(): True/False/None(unknown)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _register_edge(self, edge: Edge):
        if edge.name in self._edges:
            raise ValueError(f"duplicate edge name {edge.name!r}")
        self._edges[edge.name] = edge
        self._vertices.setdefault(edge.tail, None)
        self._vertices.setdefault(edge.head, None)

    def _add_element(self, element):
        self._elements.append(element)
        for e in element.edges():
            self._register_edge(e)
        return element

    def add_capacitor(self, name, tail, head, C=None):
        return self._add_element(Capacitor(name, tail, head, C))

    def add_inductor(self, name, tail, head, L=None):
        return self._add_element(Inductor(name, tail, head, L))

    def add_josephson(self, name, tail, head, EJ=None, winding=1):
        return self._add_element(JosephsonJunction(name, tail, head, EJ, winding))

    def add_qps(self, name, tail, head, ES=None, winding=1):
        return self._add_element(QuantumPhaseSlip(name, tail, head, ES, winding))

    def add_gyrator(self, edge1, edge2, G=None):
        """Add an ideal gyrator coupling two edges.

        Parameters
        ----------
        edge1, edge2 : tuple ``(name, tail, head)``
            The ordered pair of gyrative half-edges (e1, e2).
        G : symbol / number / str, optional
            The gyration ratio.
        """
        (n1, t1, h1) = edge1
        (n2, t2, h2) = edge2
        gyr = Gyrator(n1, t1, h1, n2, t2, h2, G)
        self._add_element(gyr)
        self._gyrator_pairs.append((n1, n2, gyr.G))
        return gyr

    def add_loop(self, name, signed_edges: Sequence[str]):
        """Declare a loop (face) as a list like ``["+e1", "-e4", "+e5"]``.

        A leading ``+`` may be omitted.  The signs are exactly B[l, e].
        """
        entries: List[Tuple[int, str]] = []
        for token in signed_edges:
            token = token.strip()
            if token.startswith("-"):
                sign, ename = -1, token[1:]
            elif token.startswith("+"):
                sign, ename = +1, token[1:]
            else:
                sign, ename = +1, token
            if ename not in self._edges:
                raise ValueError(f"loop {name!r} references unknown edge {ename!r}")
            entries.append((sign, ename))
        self._loops[name] = entries

    def infer_loops(self, force: bool = False):
        """Derive the loops automatically from the circuit graph.

        Finds a planar embedding and traces its faces (so ``dual`` and
        ``schematic`` work); for a non-planar circuit it falls back to a cycle
        basis (the Hamiltonian still works) and warns.  Declaring loops by hand
        is therefore optional.  Does nothing if loops already exist unless
        *force* is set.  Returns the list of loop names.
        """
        from .topology import infer_loops as _infer
        if self._loops and not force:
            return self.loops
        if force:
            self._loops.clear()
        loops, planar = _infer(self)
        self._planar = planar
        for name, tokens in loops:
            self.add_loop(name, tokens)
        return self.loops

    # ------------------------------------------------------------------
    # external biases (nonzero Noether constants of the manuscript)
    # ------------------------------------------------------------------
    def set_flux_bias(self, loop, value=None):
        """Thread an external flux through *loop* (a face).

        This is the loop's nonzero Noether constant: the branch fluxes around
        the loop now sum to the external flux instead of zero.  *value* is a
        symbol, number or string; if omitted a symbol ``phi_ext_<loop>`` is
        used.  The symbol equals the physical loop flux (period ``2*pi``; a
        fluxonium loop's sweet spot is at ``pi``).  Returns the symbol/value.
        """
        if loop not in self._loops:
            raise ValueError(f"unknown loop {loop!r}")
        val = sp.Symbol(f"phi_ext_{loop}") if value is None else sp.sympify(value)
        self._flux_bias[loop] = val
        return val

    def set_offset_charge(self, node, value=None):
        """Put an external (gate/offset) charge on *node* -- the LCG dual of an
        external loop flux.

        *value* is a symbol, number or string; if omitted a symbol
        ``n_g_<node>`` is used.  It enters as ``(n - n_g)`` in the node's
        charging energy (period ``1`` in Cooper-pair number).  Returns the
        symbol/value.
        """
        if node not in self._vertices:
            raise ValueError(f"unknown node {node!r}")
        val = sp.Symbol(f"n_g_{node}") if value is None else sp.sympify(value)
        self._offset_charge[node] = val
        return val

    # ------------------------------------------------------------------
    # index helpers
    # ------------------------------------------------------------------
    @property
    def vertices(self) -> List[str]:
        return list(self._vertices.keys())

    @property
    def edges(self) -> List[str]:
        return list(self._edges.keys())

    @property
    def loops(self) -> List[str]:
        return list(self._loops.keys())

    def _vidx(self):
        return {v: i for i, v in enumerate(self.vertices)}

    def _eidx(self):
        return {e: i for i, e in enumerate(self.edges)}

    def _lidx(self):
        return {l: i for i, l in enumerate(self.loops)}

    # ------------------------------------------------------------------
    # boundary matrices
    # ------------------------------------------------------------------
    def validate(self) -> bool:
        """Check the circuit is well-formed; raise ``ValueError`` otherwise.

        Verifies there are vertices and edges, that every gyrator retains both
        of its half-edges, and that each declared loop is a genuine cycle --
        i.e. Kirchhoff exactness ``B * A = 0`` holds.  A non-zero row of
        ``B * A`` means that loop's signed edge list does not close up.
        """
        if not self.vertices:
            raise ValueError("circuit has no vertices")
        if not self.edges:
            raise ValueError("circuit has no edges; add elements before reducing")
        for el in self._elements:
            if type(el).__name__ == "Gyrator":
                for e in (el.edge1, el.edge2):
                    if e.name not in self._edges:
                        raise ValueError(
                            f"gyrator {el.name!r} is missing half-edge {e.name!r}")
        A = self.incidence_matrix()
        B = self.orientation_matrix()
        BA = B * A
        if BA != sp.zeros(*BA.shape):
            for li, lname in enumerate(self.loops):
                if any(BA[li, j] != 0 for j in range(BA.cols)):
                    raise ValueError(
                        f"loop {lname!r} is not a closed cycle (Kirchhoff "
                        f"exactness B*A = 0 fails on its row); check the signed "
                        f"edge list -- the edges must form a closed loop")
            raise ValueError("Kirchhoff exactness B*A = 0 is violated")
        return True

    def incidence_matrix(self) -> sp.Matrix:
        """A : |E| x |V|, with A[e, v] = +1 (toward v), -1 (away), 0 else."""
        vidx = self._vidx()
        A = sp.zeros(len(self.edges), len(self.vertices))
        for ei, ename in enumerate(self.edges):
            edge = self._edges[ename]
            A[ei, vidx[edge.head]] += 1
            A[ei, vidx[edge.tail]] += -1
        return A

    def orientation_matrix(self) -> sp.Matrix:
        """B : |F| x |E|, with B[l, e] = +1 / -1 / 0 (see add_loop)."""
        eidx = self._eidx()
        B = sp.zeros(len(self.loops), len(self.edges))
        for li, lname in enumerate(self.loops):
            for sign, ename in self._loops[lname]:
                B[li, eidx[ename]] += sign
        return B

    def _class_projectors(self):
        """Diagonal projectors P_C, P_I, P_G onto each edge class (|E| x |E|)."""
        n = len(self.edges)
        PC, PI, PG = sp.zeros(n), sp.zeros(n), sp.zeros(n)
        for ei, ename in enumerate(self.edges):
            cls = self._edges[ename].edge_class
            if cls == CAPACITIVE:
                PC[ei, ei] = 1
            elif cls == INDUCTIVE:
                PI[ei, ei] = 1
            else:
                PG[ei, ei] = 1
        return PC, PI, PG

    def connection_matrix(self) -> sp.Matrix:
        """M = (1/2) B (P_C - P_I) A   ->   |F| x |V|."""
        A = self.incidence_matrix()
        B = self.orientation_matrix()
        PC, PI, _ = self._class_projectors()
        return sp.Rational(1, 2) * B * (PC - PI) * A

    def _gamma(self, ename1: str, ename2: str) -> sp.Matrix:
        """Antisymmetric form Gamma_g = |e1><e2| - |e2><e1|  (|E| x |E|)."""
        eidx = self._eidx()
        n = len(self.edges)
        G = sp.zeros(n)
        i, j = eidx[ename1], eidx[ename2]
        G[i, j] += 1
        G[j, i] += -1
        return G

    def omega(self) -> sp.Matrix:
        """The antisymmetric form Omega of Eq. (omega), size (|V|+|F|)^2."""
        A = self.incidence_matrix()
        B = self.orientation_matrix()
        M = self.connection_matrix()
        nV, nF = len(self.vertices), len(self.loops)

        top_left = sp.zeros(nV, nV)
        bot_right = sp.zeros(nF, nF)
        for (n1, n2, G) in self._gyrator_pairs:
            Gam = self._gamma(n1, n2)
            top_left += G * (A.T * Gam * A)
            bot_right += -(1 / G) * (B * Gam * B.T)

        Omega = sp.zeros(nV + nF, nV + nF)
        Omega[:nV, :nV] = top_left
        Omega[:nV, nV:] = -2 * M.T
        Omega[nV:, :nV] = 2 * M
        Omega[nV:, nV:] = bot_right
        return sp.simplify(Omega)

    # ------------------------------------------------------------------
    # coordinate symbols
    # ------------------------------------------------------------------
    def coordinate_symbols(self):
        """Return ``(phi, q, phidot, qdot)`` as lists of sympy symbols."""
        phi = [sp.Symbol(f"phi_{v}") for v in self.vertices]
        q = [sp.Symbol(f"q_{l}") for l in self.loops]
        phidot = [sp.Symbol(f"\\dot{{\\phi}}_{v}") for v in self.vertices]
        qdot = [sp.Symbol(f"\\dot{{q}}_{l}") for l in self.loops]
        return phi, q, phidot, qdot

    def edge_flux(self, ename, phi=None):
        """Phi_e = sum_v A[e, v] phi_v."""
        if phi is None:
            phi, _, _, _ = self.coordinate_symbols()
        A = self.incidence_matrix()
        ei = self._eidx()[ename]
        return sp.expand(sum(A[ei, vi] * phi[vi] for vi in range(len(self.vertices))))

    def edge_charge(self, ename, q=None):
        """Q_e = sum_l q_l B[l, e]."""
        if q is None:
            _, q, _, _ = self.coordinate_symbols()
        B = self.orientation_matrix()
        ei = self._eidx()[ename]
        return sp.expand(sum(q[li] * B[li, ei] for li in range(len(self.loops))))

    # ------------------------------------------------------------------
    # Lagrangian and energy
    # ------------------------------------------------------------------
    def energy(self, phi=None, q=None) -> sp.Expr:
        """E = sum_{e in C} E^C_e(Q_e) + sum_{e in I} E^I_e(Phi_e).

        External biases enter here as constant offsets to the edge variables: an
        external flux through a loop is added to that loop's inductive edge
        fluxes (weighted by ``B`` and split evenly so the loop integral equals
        the bias symbol), and an offset charge on a node is added to that node's
        capacitive edge charges (weighted by ``A``, split evenly).  This is the
        manuscript's nonzero-Noether-constant prescription; with no bias set the
        energy is unchanged.
        """
        if phi is None or q is None:
            phi, q, _, _ = self.coordinate_symbols()
        A = self.incidence_matrix()
        B = self.orientation_matrix()
        eidx, vidx, lidx = self._eidx(), self._vidx(), self._lidx()

        # how many edges of the biased class touch each biased loop / node
        ind_on_loop = {l: sum(1 for e, ed in self._edges.items()
                              if ed.edge_class == INDUCTIVE and B[lidx[l], eidx[e]] != 0)
                       for l in self._flux_bias}
        cap_at_node = {v: sum(1 for e, ed in self._edges.items()
                              if ed.edge_class == CAPACITIVE and A[eidx[e], vidx[v]] != 0)
                       for v in self._offset_charge}

        E = sp.Integer(0)
        for ename, edge in self._edges.items():
            ei = eidx[ename]
            if edge.edge_class == CAPACITIVE:
                Q = self.edge_charge(ename, q)
                for node, ng in self._offset_charge.items():
                    if cap_at_node[node]:
                        Q += A[ei, vidx[node]] * ng / cap_at_node[node]
                E += edge.energy(Q)
            elif edge.edge_class == INDUCTIVE:
                Phi = self.edge_flux(ename, phi)
                for loop, fx in self._flux_bias.items():
                    if ind_on_loop[loop]:
                        Phi += B[lidx[loop], ei] * fx / ind_on_loop[loop]
                E += edge.energy(Phi)
        return sp.expand_trig(sp.expand(E))

    def lagrangian(self) -> sp.Expr:
        """The full flux-charge symmetric Lagrangian, Eq. (no_omega_lagrangian).

            L = <q|M|phidot> + sum_g L_g - E
        """
        phi, q, phidot, qdot = self.coordinate_symbols()
        M = self.connection_matrix()
        nV, nF = len(self.vertices), len(self.loops)

        # reciprocal symplectic term  q^T M phidot
        sympl = sp.Integer(0)
        for li in range(nF):
            for vi in range(nV):
                if M[li, vi] != 0:
                    sympl += q[li] * M[li, vi] * phidot[vi]

        # gyrator terms  (G/2) Phi_e1 dPhi_e2  -  (1/2G) Q_e1 dQ_e2
        for (n1, n2, G) in self._gyrator_pairs:
            Phi1 = self.edge_flux(n1, phi)
            dPhi2 = self.edge_flux(n2, phidot)
            Q1 = self.edge_charge(n1, q)
            dQ2 = self.edge_charge(n2, qdot)
            sympl += G / 2 * Phi1 * dPhi2 - 1 / (2 * G) * Q1 * dQ2

        return sp.expand(sympl) - self.energy(phi, q)

    # ------------------------------------------------------------------
    # reduction to a Hamiltonian
    # ------------------------------------------------------------------
    def hamiltonian(self, ground=None, open_loops=None,
                    keep=None, eliminate=None, strict=True, canonical=False):
        """Reduce the circuit to its Hamiltonian.

        The reduction follows Section "Constraints and Quantization" of
        Salcedo et al.: the null vectors of ``Omega`` and the symmetries of the
        energy are classified into *null-vector* and *Noether* constraints,
        which together with the gauge choices and the cyclic coordinates are
        solved to produce the Hamiltonian on the reduced phase space.

        Parameters
        ----------
        ground : str, optional
            Node whose flux is set to zero (global-flux gauge).  Defaults to
            the first declared vertex.  Pass ``ground=False`` to skip.
        open_loops : str or sequence of str, optional
            Loop(s) whose charge is set to zero (global-charge gauge).  No
            loop is opened by default; for a planar circuit the outer face is
            the usual choice.
        keep : sequence, optional
            Coordinates (symbols or strings) that must *not* be eliminated when
            solving the constraints.
        eliminate : sequence, optional
            Preferred coordinates to solve the constraints for.
        strict : bool, optional
            If true (the default) raise when the reduction is incomplete -- the
            reduced symplectic form is degenerate, meaning constraints remain
            that this routine did not resolve (for example a non-default gauge
            is needed).  Set ``strict=False`` to return the partial result and
            inspect ``ReductionResult.complete`` yourself.

        Returns
        -------
        ReductionResult
            Carries ``.H`` (the Hamiltonian) along with the typed constraints,
            gauge choices, cyclic coordinates, conjugate pairs, and the
            ``.complete`` flag.

        Notes
        -----
        The energy, symplectic structure and constraints are built
        automatically and exactly.  The *gauge* (ground node, open loop) is a
        physical choice supplied here, and the coordinate each constraint is
        solved for is reported (and steerable via ``keep`` / ``eliminate``).
        Inspect :meth:`ReductionResult.report` to see every step, and use
        :class:`~fluxcharge.reduction.Reducer` directly for full manual control.
        """
        from .reduction import Reducer

        if not self._loops:
            self.infer_loops()      # loops are optional: derive them from the graph
        self.validate()

        r = Reducer(self)
        if ground is None:
            if self.vertices:
                r.ground(self.vertices[0])
        elif ground is not False:
            r.ground(ground)
        if open_loops:
            if isinstance(open_loops, str):
                open_loops = [open_loops]
            r.open_loop(*open_loops)
        r.auto_constraints(keep=keep, eliminate=eliminate)
        result = r.to_hamiltonian()

        if strict and not result.complete:
            residual = ", ".join(map(str, result.coordinates))
            raise RuntimeError(
                "reduction is incomplete: the reduced symplectic form is "
                f"degenerate on the surviving coordinates ({residual}). "
                "Unresolved constraints remain -- this usually means a gauge "
                "must be chosen (pass ground=/open_loops=) or a constraint "
                "target steered (keep=/eliminate=). Re-run with strict=False "
                "to inspect the partial ReductionResult and its .report()."
            )
        if canonical:
            result = result.canonical()
        return result

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    @property
    def parameters(self):
        syms = set()
        for el in self._elements:
            syms |= el.parameters
        for val in list(self._flux_bias.values()) + list(self._offset_charge.values()):
            syms |= val.free_symbols
        return syms

    def natural_params(self, physical):
        """Convert physical element values (fF / nH / GHz) into the numeric
        parameters for diagonalization, so eigenvalues come out in GHz.

        ``physical`` maps each parameter symbol to a value with units, e.g.
        ``{"C": "70fF", "E_J": "15GHz"}``.  See :func:`fluxcharge.units.to_natural`.
        """
        from .units import to_natural
        return to_natural(self, physical)

    def summary(self) -> str:
        lines = [
            f"Circuit: {len(self.vertices)} vertices, "
            f"{len(self.edges)} edges, {len(self.loops)} loops",
        ]
        for ename, e in self._edges.items():
            lines.append(f"  edge {ename:>6}: {e.tail} -> {e.head}  [{e.edge_class}]")
        for lname, ents in self._loops.items():
            txt = " ".join(("+" if s > 0 else "-") + n for s, n in ents)
            lines.append(f"  loop {lname:>6}: {txt}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # visualization
    # ------------------------------------------------------------------
    def to_networkx(self):
        """Reconstruct a :class:`networkx.MultiDiGraph` from ``A`` and ``B``.

        Requires the optional ``networkx`` dependency.  See
        :func:`fluxcharge.visualize.circuit_to_networkx`.
        """
        from .visualize import circuit_to_networkx
        return circuit_to_networkx(self)

    def draw(self, path=None, layout="auto", show_loops=True, title=None,
             ax=None, dpi=160, **kwargs):
        """Draw the circuit *topology* graph; optionally save to *path*.

        This is the networkx view (nodes and oriented edges).  For a
        lumped-element **schematic** with real circuit symbols and straight
        wires, use :meth:`schematic` instead.

        Requires the optional ``networkx`` and ``matplotlib`` dependencies.
        Returns the matplotlib ``Axes``.
        """
        from .visualize import draw_circuit
        ax = draw_circuit(self, ax=ax, layout=layout, show_loops=show_loops,
                          title=title, **kwargs)
        if path is not None:
            ax.figure.savefig(path, dpi=dpi, bbox_inches="tight")
        return ax

    def schematic(self, path=None, layout="auto", positions=None,
                  outer_loop=None, show_values=True, **kwargs):
        """Draw the circuit as a planar lumped-element schematic and optionally
        save it.

        Renders real circuit symbols (inductor coil, capacitor plates, the
        Josephson boxed-X, a charge-dual symbol for the quantum phase slip, and
        gyrator half-edges) with straight wires, using ``schemdraw``.  Node
        placement uses the planar face structure encoded in ``B``: the outer
        face is placed on a convex polygon and interior nodes by the barycentric
        (Tutte) method, giving a crossing-free drawing.  Edges on the outer face
        are routed around the outside so their interior parallel siblings stay
        straight.

        Requires the optional ``schemdraw`` dependency
        (``pip install "fluxcharge[schematic]"``).  Returns the
        ``schemdraw.Drawing``.

        Parameters
        ----------
        outer_loop : str, optional
            Which declared loop is the outer face.  Defaults to the largest
            loop (the loop you open for the global-charge gauge is usually the
            right choice).
        positions : dict, optional
            Explicit ``{node: (x, y)}`` placement, overriding the layout -- the
            natural hook for an interactive editor.
        """
        from .schematic import draw_schematic
        return draw_schematic(self, file=path, layout=layout, positions=positions,
                              outer_loop=outer_loop, show_values=show_values,
                              **kwargs)

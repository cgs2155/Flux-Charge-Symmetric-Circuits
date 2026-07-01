"""
Branch currents and node/element voltages as operators, derived from the
reduced Hamiltonian and the circuit graph.

For an edge ``e`` with branch flux ``Phi_e`` and branch charge ``Q_e``, voltage
and current are the rates of those branch variables:

    V_e = d Phi_e / dt ,      I_e = d Q_e / dt .

There are two routes, both derived directly from ``H``:

* **Constitutive (algebraic).** The element's own current/voltage law, i.e. the
  derivative of *its* energy with respect to its branch variable::

      inductor    I = dE/dPhi = Phi_e / L        Josephson  I = E_J sin(Phi_e)
      capacitor   V = dE/dQ   = Q_e / C          QPS        V = E_S sin(Q_e)

* **Equation of motion.** For the complementary quantities (voltage across an
  inductive element, current through a capacitive one) and for the voltage
  between two nodes, ``X_dot = {X, H}`` via the reduced symplectic bracket
  ``f^{-1}`` -- the same structure :meth:`ReductionResult.commutators` reports
  (calibrated so the transmon has ``[phi, q] = i hbar``).  The EOM route
  reproduces the constitutive one where they overlap (a built-in check: for an
  LC loop the inductor and capacitor currents are equal and opposite).

Branch variables include any external-bias offset exactly as
:meth:`Circuit.energy` injects it, so a flux-biased circuit's Josephson current
uses the biased phase ``E_J sin(phi_ext/2 + phi)``.

Units: results are in the manuscript's natural units (``hbar = 1``, ``G_0 = 1``);
convert to physical amperes / volts through :mod:`fluxcharge.units`.
"""
from __future__ import annotations

import sympy as sp

from .elements import INDUCTIVE, CAPACITIVE

_INDUCTIVE = ("Inductor", "JosephsonJunction")
_CAPACITIVE = ("Capacitor", "QuantumPhaseSlip")


def _branch_flux(circuit, edge):
    """``Phi_e`` including its share of any loop flux bias (mirrors energy())."""
    B = circuit.orientation_matrix()
    eidx, lidx = circuit._eidx(), circuit._lidx()
    ei = eidx[edge]
    Phi = circuit.edge_flux(edge)
    for loop, fx in circuit._flux_bias.items():
        n = sum(1 for x, ed in circuit._edges.items()
                if ed.edge_class == INDUCTIVE and B[lidx[loop], eidx[x]] != 0)
        if n:
            Phi += B[lidx[loop], ei] * fx / n
    return sp.expand(Phi)


def _branch_charge(circuit, edge):
    """``Q_e`` including its share of any node offset charge (mirrors energy())."""
    A = circuit.incidence_matrix()
    eidx, vidx = circuit._eidx(), circuit._vidx()
    ei = eidx[edge]
    Q = circuit.edge_charge(edge)
    for node, ng in circuit._offset_charge.items():
        n = sum(1 for x, ed in circuit._edges.items()
                if ed.edge_class == CAPACITIVE and A[eidx[x], vidx[node]] != 0)
        if n:
            Q += A[ei, vidx[node]] * ng / n
    return sp.expand(Q)


def _to_reduced(result, expr):
    """Rewrite *expr* (in original node fluxes / loop charges) in the surviving
    coordinates, applying the reduction's eliminations, gauge and dropped cyclic
    coordinates."""
    elim = {k: sp.expand(v) for k, v in result.eliminated.items()}
    for _ in range(len(elim) + 1):                    # resolve cross-references
        nxt = {k: sp.expand(v.subs(elim)) for k, v in elim.items()}
        if nxt == elim:
            break
        elim = nxt
    sub = dict(elim)
    sub.update({g: 0 for g in result.gauge})
    sub.update({c: 0 for c in result.cyclic})
    out = sp.expand(expr.subs(sub))
    # if the result was canonicalized, its surviving symbols were rescaled in
    # place (H.subs(s, s/cs)); apply the same map so the branch variable, H and
    # the symplectic form all live in one frame.
    if result.rescaling:
        out = sp.expand(out.subs({s: s / cs for s, cs in result.rescaling.items()}))
    return out


def _dot(result, X):
    """The Heisenberg rate ``{X, H} = grad(X)^T f^{-1} grad(H)`` in the reduced
    coordinates -- the same bracket ``commutators()`` reports."""
    if result.symplectic_matrix is None or not result.complete:
        raise ValueError(
            "equation-of-motion observables need a complete reduction (a "
            "non-degenerate reduced symplectic form); this reduction is incomplete.")
    coords = list(result.coordinates)
    Sinv = sp.Matrix(result.symplectic_matrix).inv()
    gX = sp.Matrix([[sp.diff(X, c) for c in coords]])
    gH = sp.Matrix([sp.diff(result.H, c) for c in coords])
    return sp.expand((gX * Sinv * gH)[0, 0])


def _element(result, edge):
    circuit = getattr(result, "circuit", None)
    if circuit is None:
        raise ValueError("this ReductionResult has no attached circuit; build it "
                         "via Circuit.hamiltonian() to use current()/voltage()")
    if edge not in circuit._edges:
        raise ValueError(f"unknown edge {edge!r}; edges: {list(circuit._edges)}")
    return circuit, circuit._edges[edge]


def current(result, edge):
    """Symbolic current operator through the element on *edge*, in the reduced
    coordinates and the manuscript's natural units.

    Inductive elements use the constitutive law ``dE/dPhi`` (``Phi/L`` for an
    inductor, ``E_J sin Phi`` for a junction); capacitive elements use the
    displacement current ``d Q_e / dt = {Q_e, H}``.
    """
    circuit, ed = _element(result, edge)
    et = type(ed.element).__name__
    if et in _INDUCTIVE:
        Phi = _to_reduced(result, _branch_flux(circuit, edge))
        x = sp.Dummy("x")
        dEdPhi = sp.diff(ed.energy(x), x)
        return sp.expand(dEdPhi.subs(x, Phi))
    if et in _CAPACITIVE:
        # displacement current d Q_e/dt = {Q_e, H}; the sign aligns the branch
        # charge's orientation with the flux (constitutive) convention, so
        # Kirchhoff's current law closes across every node.
        Q = _to_reduced(result, _branch_charge(circuit, edge))
        return -_dot(result, Q)
    raise ValueError(f"edge {edge!r} carries no one-port element ({et})")


def voltage(result, node_or_edge, node_b=None):
    """Symbolic voltage operator, in the reduced coordinates and natural units.

    * ``voltage(result, a, b)`` -- the voltage between nodes *a* and *b*,
      ``V_a - V_b = {phi_a - phi_b, H}``.
    * ``voltage(result, edge)`` -- the voltage across the element on *edge*:
      the constitutive ``dE/dQ`` for a capacitive element (``Q/C``,
      ``E_S sin Q``), or ``{Phi_e, H}`` for an inductive one.
    """
    circuit = getattr(result, "circuit", None)
    if circuit is None:
        raise ValueError("this ReductionResult has no attached circuit; build it "
                         "via Circuit.hamiltonian() to use current()/voltage()")
    if node_b is not None:                             # node-to-node voltage
        phi, _q, _, _ = circuit.coordinate_symbols()
        vidx = circuit._vidx()
        for n in (node_or_edge, node_b):
            if n not in vidx:
                raise ValueError(f"unknown node {n!r}; nodes: {list(vidx)}")
        dphi = _to_reduced(result, phi[vidx[node_or_edge]] - phi[vidx[node_b]])
        return _dot(result, dphi)

    circuit, ed = _element(result, node_or_edge)
    et = type(ed.element).__name__
    if et in _CAPACITIVE:
        Q = _to_reduced(result, _branch_charge(circuit, node_or_edge))
        x = sp.Dummy("x")
        dEdQ = sp.diff(ed.energy(x), x)
        return sp.expand(dEdQ.subs(x, Q))
    if et in _INDUCTIVE:
        # voltage across = d Phi_e/dt = {Phi_e, H}; the sign is the dual of the
        # capacitive displacement current, so Kirchhoff's voltage law closes
        # around every loop.
        Phi = _to_reduced(result, _branch_flux(circuit, node_or_edge))
        return -_dot(result, Phi)
    raise ValueError(f"edge {node_or_edge!r} carries no one-port element ({et})")

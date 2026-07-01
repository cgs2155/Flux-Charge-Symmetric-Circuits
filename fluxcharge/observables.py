"""
Branch currents and node/element voltages, solved from the reduced state by
circuit analysis (constitutive laws + Kirchhoff), not from Heisenberg brackets.

Why a circuit solve rather than ``{X, H}``:  the reduction eliminates some
coordinates (and ``canonical()`` rescales the survivors), so the bracket
velocity ``{phi_v, H}`` is *not* the physical node voltage once a gyrator mixes
the flux and charge sectors -- it disagrees with the constitutive ``V = Q/C``,
which is definitional.  Instead we read the branch fluxes ``Phi_e`` and charges
``Q_e`` off the reduced state and solve the instantaneous circuit:

* **Constitutive laws** give the primary quantity of each element exactly --
  inductive **current** ``dE/dPhi`` (``Phi/L``, ``E_J sin Phi``) and capacitive
  **voltage** ``dE/dQ`` (``Q/C``, ``E_S sin Q``).  Branch variables include any
  external-bias offset (as ``Circuit.energy`` injects it), so a flux-biased
  junction's current carries the biased phase.
* **Node potentials** ``u_v`` follow from the capacitor voltages
  (``u_head - u_tail = dE/dQ``); the voltage across any edge is ``u_head - u_tail``.
* **Gyrator** half-edge current is set by its partner's voltage,
  ``I_1 = -G V_2``, ``I_2 = +G V_1`` -- the ideal-gyrator relation.
* **Kirchhoff's current law** (``sum_e A[e,n] I_e = 0``) then fixes the
  remaining (capacitive displacement) currents.

The result satisfies KCL at every node and KVL around every loop by
construction, and works for non-reciprocal (gyrator) circuits.  Units are the
manuscript's natural units (``hbar = 1``, ``G_0 = 1``); convert with
:mod:`fluxcharge.units`.
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
    coordinates, applying the eliminations, gauge, dropped cyclic coordinates
    and any canonical rescaling."""
    elim = {k: sp.expand(v) for k, v in result.eliminated.items()}
    for _ in range(len(elim) + 1):
        nxt = {k: sp.expand(v.subs(elim)) for k, v in elim.items()}
        if nxt == elim:
            break
        elim = nxt
    sub = dict(elim)
    sub.update({g: 0 for g in result.gauge})
    sub.update({c: 0 for c in result.cyclic})
    out = sp.expand(expr.subs(sub))
    if result.rescaling:
        out = sp.expand(out.subs({s: s / cs for s, cs in result.rescaling.items()}))
    return out


def _dEd(edge, branch_var):
    """Constitutive derivative dE/d(branch var): the inductive current-phase law
    for an inductive edge, or the capacitive voltage-charge law."""
    x = sp.Dummy("x")
    return sp.expand(sp.diff(edge.energy(x), x).subs(x, branch_var))


def _solve_circuit(result):
    """Solve the instantaneous circuit for node potentials and every branch
    current/voltage from the reduced state.  Returns ``(u, current, voltage)``:
    ``u`` maps node -> potential, ``current``/``voltage`` map edge -> operator.
    All are SymPy expressions in the reduced coordinates (natural units)."""
    circuit = getattr(result, "circuit", None)
    if circuit is None:
        raise ValueError("this ReductionResult has no attached circuit; build it "
                         "via Circuit.hamiltonian() to use current()/voltage()")
    edges = list(circuit.edges)
    A = circuit.incidence_matrix()
    eidx, vidx = circuit._eidx(), circuit._vidx()
    nodes = list(circuit.vertices)

    Q = {e: _to_reduced(result, _branch_charge(circuit, e)) for e in edges}
    Phi = {e: _to_reduced(result, _branch_flux(circuit, e)) for e in edges}

    # node potentials u_v (pick a reference node = 0; only differences matter)
    ref = (circuit.ground if getattr(circuit, "ground", None) in vidx else nodes[0])
    u = {v: (sp.Integer(0) if v == ref else sp.Symbol(f"_u_{v}")) for v in nodes}

    def across(e):
        ed = circuit._edges[e]
        return u[ed.head] - u[ed.tail]

    # capacitor voltages fix the node potentials: u_head - u_tail = dE/dQ
    cap_eqs = [sp.Eq(across(e), _dEd(circuit._edges[e], Q[e]))
               for e in edges if type(circuit._edges[e].element).__name__ in _CAPACITIVE]
    unknown_u = [u[v] for v in nodes if v != ref]
    if unknown_u:
        sol = sp.solve(cap_eqs, unknown_u, dict=True)
        if not sol:
            raise ValueError(
                "cannot determine the node potentials from the capacitor "
                "voltages -- a node has no capacitive path to the reference "
                "(its potential is not fixed by the static constitutive laws). "
                "Currents/voltages for such a purely inductive island are not "
                "supported.")
        u = {v: (val.subs(sol[0]) if hasattr(val, "subs") else val)
             for v, val in u.items()}

    voltage = {e: sp.expand(across(e)) for e in edges}

    # gyrator half-edge current = -/+ G * partner voltage (ideal-gyrator relation)
    gyr_rel = {}
    for el in circuit._elements:
        if type(el).__name__ == "Gyrator":
            gyr_rel[el.edge1.name] = (el.edge2.name, -el.G)
            gyr_rel[el.edge2.name] = (el.edge1.name, el.G)

    known_I, cap_I = {}, {}
    for e in edges:
        et = type(circuit._edges[e].element).__name__
        if et in _INDUCTIVE:
            known_I[e] = _dEd(circuit._edges[e], Phi[e])       # constitutive current
        elif et == "Gyrator":
            pnm, g = gyr_rel[e]
            known_I[e] = sp.expand(g * voltage[pnm])
        else:                                                  # capacitive: unknown
            cap_I[e] = sp.Symbol(f"_I_{e}")

    current = dict(known_I)
    if cap_I:
        def I(e):
            return current[e] if e in current else cap_I[e]
        kcl = [sum(A[eidx[e], vidx[v]] * I(e) for e in edges) for v in nodes]
        sol = sp.solve(kcl, list(cap_I.values()), dict=True)
        if not sol:
            raise ValueError("Kirchhoff's current law did not determine the "
                             "capacitive currents (circuit under/over-constrained).")
        current.update({e: sp.expand(sym.subs(sol[0])) for e, sym in cap_I.items()})

    return u, current, voltage


def current(result, edge):
    """Symbolic current operator through the element on *edge*, in the reduced
    coordinates and natural units (see the module docstring)."""
    circuit = getattr(result, "circuit", None)
    if circuit is None or edge not in circuit._edges:
        raise ValueError(f"unknown edge {edge!r}")
    _u, I, _v = _solve_circuit(result)
    return I[edge]


def voltage(result, node_or_edge, node_b=None):
    """Symbolic voltage operator: ``voltage(result, a, b)`` is the node-to-node
    voltage ``V_a - V_b``; ``voltage(result, edge)`` is the voltage across the
    element on *edge*."""
    circuit = getattr(result, "circuit", None)
    if circuit is None:
        raise ValueError("this ReductionResult has no attached circuit")
    u, _I, V = _solve_circuit(result)
    if node_b is not None:
        for n in (node_or_edge, node_b):
            if n not in u:
                raise ValueError(f"unknown node {n!r}; nodes: {list(u)}")
        return sp.expand(u[node_or_edge] - u[node_b])
    if node_or_edge not in V:
        raise ValueError(f"unknown edge {node_or_edge!r}")
    return V[node_or_edge]

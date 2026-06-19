"""
Canonical transformations on LCG circuits.

Currently this implements the **LCG duality transform** of Salcedo, Cocquyt,
Osborne & Houck, Sec. "Circuit Duality":

    phi_v -> q_l,   q_l -> phi_v          (flux <-> charge)
    A     -> B^T,   B   -> A^T            (vertices <-> faces)
    C <-> I   (capacitor <-> inductor),   JJ <-> QPS
    gyrator edge pair preserved,  G -> -1/G

Concretely the dual circuit has

* one vertex per face (loop) of the original,
* the same edge set, each edge now joining the two faces it bordered
  (``A*[e, l] = B[l, e]``: head = the face with ``B = +1``, tail the ``-1`` one),
* one loop per vertex of the original (``B*[v, e] = A[e, v]``: the signed list
  of edges incident on ``v``),

and per-element ``C* = L``, ``L* = C``, ``E_J* = E_S``, ``E_S* = E_J`` (scalar
value preserved) and ``G* = -1/G``.  Because ``B* A* = (B A)^T = 0``, the dual
automatically satisfies Kirchhoff exactness, and the transform is an involution
up to a global edge-orientation reversal.  It preserves the Hamiltonian
spectrum.
"""

from __future__ import annotations

import sympy as sp

from .circuit import Circuit
from .elements import (
    Capacitor, Inductor, JosephsonJunction, QuantumPhaseSlip, Gyrator,
)


def dual(circuit: Circuit) -> Circuit:
    """Return the LCG dual of *circuit* (a new :class:`Circuit`)."""
    circuit.validate()

    # B[l, e]: which faces each edge borders, with sign
    edge_faces = {e: {} for e in circuit.edges}
    for loop, entries in circuit._loops.items():
        for sign, ename in entries:
            edge_faces[ename][loop] = sign

    def dual_ends(ename):
        faces = edge_faces[ename]
        plus = [l for l, s in faces.items() if s == 1]
        minus = [l for l, s in faces.items() if s == -1]
        if len(plus) != 1 or len(minus) != 1:
            raise ValueError(
                f"edge {ename!r} borders faces {faces}; to dualize, every edge must "
                "lie on exactly two declared faces (one +1, one -1). Declare all "
                "faces of the planar embedding, including the outer face.")
        return minus[0], plus[0]      # (tail*, head*)  since A*[e,l] = B[l,e]

    D = Circuit()
    title = getattr(circuit, "title", None)
    D.title = f"dual of {title}" if title else "dual circuit"
    D.ground = None
    D.open_loops = []

    # reciprocal elements: swap class, keep scalar value, move to dual endpoints
    for elem in circuit._elements:
        if isinstance(elem, Gyrator):
            continue
        e = elem.edges()[0]
        t, h = dual_ends(e.name)
        if isinstance(elem, Capacitor):
            D.add_inductor(e.name, t, h, L=elem.C)
        elif isinstance(elem, Inductor):
            D.add_capacitor(e.name, t, h, C=elem.L)
        elif isinstance(elem, JosephsonJunction):
            D.add_qps(e.name, t, h, ES=elem.EJ)
        elif isinstance(elem, QuantumPhaseSlip):
            D.add_josephson(e.name, t, h, EJ=elem.ES)
        else:  # pragma: no cover - defensive
            raise TypeError(f"cannot dualize element {elem!r}")

    # gyrators: same ordered edge pair, dual endpoints, ratio -> -1/G
    for elem in circuit._elements:
        if not isinstance(elem, Gyrator):
            continue
        e1, e2 = elem.edge1, elem.edge2
        t1, h1 = dual_ends(e1.name)
        t2, h2 = dual_ends(e2.name)
        D.add_gyrator((e1.name, t1, h1), (e2.name, t2, h2),
                      G=sp.simplify(-1 / elem.G))

    # dual loops = original vertices: B*[v, e] = A[e, v]
    for v in circuit.vertices:
        entries = []
        for ename, edge in circuit._edges.items():
            if edge.head == v:
                entries.append("+" + ename)
            elif edge.tail == v:
                entries.append("-" + ename)
        D.add_loop(v, entries)

    return D

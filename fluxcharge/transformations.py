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

from collections import OrderedDict

import sympy as sp

from .circuit import Circuit
from .elements import (
    Capacitor, Inductor, JosephsonJunction, QuantumPhaseSlip, Gyrator,
)


def _completed_faces(circuit):
    """Declared faces, plus the outer face if it was left out.

    Dualization needs the *complete* planar embedding: every edge must border
    exactly two faces (one ``+1``, one ``-1``).  A netlist often declares only
    the inner faces (the transmon, e.g., declares one loop), so the outer face
    is missing and some edges border just one face.  In a planar embedding the
    signed face boundaries sum to zero on every edge, so the outer face is
    exactly ``-(sum of the declared faces)``; adding it is what makes every edge
    border two faces.  Because each declared face satisfies ``B*A = 0``, so does
    their (negated) sum, so the synthesized outer face is automatically a valid
    cycle.  Returns an ``OrderedDict`` ``{loop: [(sign, edge), ...]}``.
    """
    faces = OrderedDict((l, list(ents)) for l, ents in circuit._loops.items())

    residual = {e: 0 for e in circuit.edges}
    for ents in faces.values():
        for sign, ename in ents:
            residual[ename] += sign
    missing = {e: -s for e, s in residual.items() if s != 0}
    if missing:
        name = "outer"
        i = 0
        while name in faces:
            i += 1
            name = f"outer{i}"
        faces[name] = [(sign, e) for e, sign in missing.items()]

    # verify the embedding is now complete: each edge on exactly one +1 and one -1
    for ename in circuit.edges:
        plus = sum(1 for ents in faces.values() for s, e in ents if e == ename and s == 1)
        minus = sum(1 for ents in faces.values() for s, e in ents if e == ename and s == -1)
        if plus != 1 or minus != 1:
            raise ValueError(
                f"cannot complete the planar embedding for dualization: edge "
                f"{ename!r} borders {plus} face(s) with +1 and {minus} with -1 "
                "(need exactly one each). Declare the faces of a planar embedding.")
    return faces


def dual(circuit: Circuit) -> Circuit:
    """Return the LCG dual of *circuit* (a new :class:`Circuit`).

    If the netlist declared only the inner faces (so the outer face is implicit,
    as for the transmon), the outer face is completed automatically; any circuit
    that reduces to a Hamiltonian therefore also dualizes.
    """
    # loops are optional: derive them from the planar embedding if none were
    # declared (e.g. a circuit imported from scqubits).  Duality needs faces, so
    # refuse a non-planar circuit with a clear message.
    if not circuit._loops:
        circuit.infer_loops()
    if circuit._planar is False:
        raise ValueError(
            "cannot dualize a non-planar circuit: the LCG duality is the planar "
            "vertices<->faces duality, and a non-planar graph has no face "
            "structure (its Hamiltonian still reduces, but it has no dual).")
    circuit.validate()

    # B[l, e]: which faces each edge borders, with sign (outer face completed)
    faces = _completed_faces(circuit)
    edge_faces = {e: {} for e in circuit.edges}
    for loop, entries in faces.items():
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
            D.add_qps(e.name, t, h, ES=elem.EJ, winding=elem.winding)
        elif isinstance(elem, QuantumPhaseSlip):
            D.add_josephson(e.name, t, h, EJ=elem.ES, winding=elem.winding)
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

    # external biases follow the loop<->node swap: an external flux through a
    # loop becomes an offset charge on the corresponding dual node, and an
    # offset charge on a node becomes an external flux through the dual loop.
    for loop, fx in circuit._flux_bias.items():
        D.set_offset_charge(loop, fx)
    for node, ng in circuit._offset_charge.items():
        D.set_flux_bias(node, ng)

    return D


def _readd_element(dest: Circuit, el):
    """Re-add a copy of *el* (any element type) to *dest*."""
    if isinstance(el, Gyrator):
        dest.add_gyrator((el.edge1.name, el.edge1.tail, el.edge1.head),
                         (el.edge2.name, el.edge2.tail, el.edge2.head), G=el.G)
    elif isinstance(el, Capacitor):
        dest.add_capacitor(el._edge.name, el._edge.tail, el._edge.head, C=el.C)
    elif isinstance(el, Inductor):
        dest.add_inductor(el._edge.name, el._edge.tail, el._edge.head, L=el.L)
    elif isinstance(el, JosephsonJunction):
        dest.add_josephson(el._edge.name, el._edge.tail, el._edge.head,
                           EJ=el.EJ, winding=el.winding)
    elif isinstance(el, QuantumPhaseSlip):
        dest.add_qps(el._edge.name, el._edge.tail, el._edge.head,
                     ES=el.ES, winding=el.winding)
    else:  # pragma: no cover - defensive
        raise TypeError(f"cannot copy element {el!r}")


def move_across_gyrator(circuit: Circuit, element_edge: str) -> Circuit:
    """Move a reciprocal one-port across the gyrator terminating its port.

    Implements the manuscript's **partial-dual move** (Sec. "Partial Dual
    Transformations") in the Tellegen single-element case: a reciprocal one-port
    element on ``element_edge`` that *terminates* one half-edge of a gyrator is
    relocated to the gyrator's other port as its **dual**, with the gyration
    ratio carried by the conservation law ``phi_near = q_far / G``:

    * capacitor ``C``  ->  inductor ``L = C / G**2``
    * inductor ``L``   ->  capacitor ``C = G**2 * L``
    * Josephson ``E_J`` -> quantum phase slip ``E_S = E_J``
    * quantum phase slip ``E_S`` -> Josephson ``E_J = E_S``

    Because the source port then carries only the bare gyrator half-edge, the
    gyrator is removed by the open/closed-terminated deletion rule.  The
    transformation is a point transformation and so preserves the Hamiltonian
    spectrum (Tellegen: a gyrator makes a reciprocal element behave as its dual).

    Currently supports the single-element termination case: ``element_edge``
    must be the only non-gyrative element on its port.

    For a **linear** element the gyration ratio is absorbed into the dual value
    (``L = C / G**2``), so any ``G`` is fine.  For a **nonlinear** element the
    ratio lands in the cosine argument, ``cos(q / G)`` -- spectrally valid for
    any ``G`` (the dual element carries ``winding = G``), but only a *standard*
    phase slip when ``|G| = 1``; for a compact charge a non-integer ``1/G`` has
    no integer-lattice representation (the diagonalizer raises when you quantize
    it as compact, while an extended coordinate is unrestricted).  A ``|G| != 1``
    nonlinear move therefore *warns* rather than refusing.

    The move is a point transformation, so it **preserves well-posedness** (and
    ill-posedness): the gyrator-coupled element that gives the moved element its
    kinetic term on one side becomes the structure that constrains its dual on
    the other.  A well-posed junction-via-gyrator (a JJ whose port sees an
    effective capacitance, i.e. an *inductor* across the partner port) maps to a
    well-posed phase-slip-via-gyrator, with the same spectrum.  If the input is
    ill-posed -- e.g. a JJ given an effective *inductance* (a capacitor across
    the partner port), so the JJ has no kinetic term -- the output is ill-posed
    too, as it must be.
    """
    import warnings
    X = next((el for el in circuit._elements
              if getattr(el, "_edge", None) is not None
              and el._edge.name == element_edge), None)
    if X is None:
        raise ValueError(f"no one-port element on edge {element_edge!r}")
    if isinstance(X, Gyrator):
        raise ValueError(f"edge {element_edge!r} is a gyrator, not a one-port")

    port = frozenset((X._edge.tail, X._edge.head))

    # find the gyrator with a half-edge across the same port (X terminates it)
    gyr = near = far = None
    for el in circuit._elements:
        if not isinstance(el, Gyrator):
            continue
        for a, b in ((el.edge1, el.edge2), (el.edge2, el.edge1)):
            if frozenset((a.tail, a.head)) == port:
                gyr, near, far = el, a, b
    if gyr is None:
        raise ValueError(
            f"element on {element_edge!r} does not terminate a gyrator port "
            "(no gyrator half-edge shares its two nodes)")

    others = [el for el in circuit._elements
              if el is not X and el is not gyr
              and getattr(el, "_edge", None) is not None
              and frozenset((el._edge.tail, el._edge.head)) == port]
    if others:
        raise NotImplementedError(
            "the source gyrator port carries other elements "
            f"({[e._edge.name for e in others]}); only single-element "
            "termination is supported so far")

    G = gyr.G
    nonlinear = isinstance(X, (JosephsonJunction, QuantumPhaseSlip))
    if nonlinear and sp.simplify(G**2 - 1) != 0:
        warnings.warn(
            f"moving a {type(X).__name__} across a gyrator with |G| != 1 "
            f"(G={G}) produces a cos(q/G) phase slip (winding = G): spectrally "
            "valid, but a standard element only at |G| = 1. Quantized as a "
            "compact coordinate it has no integer-lattice representation unless "
            "1/G is an integer (the diagonalizer will raise); as an extended "
            "coordinate it is unrestricted.", stacklevel=2)

    D = Circuit()
    title = getattr(circuit, "title", None)
    D.title = f"{title} (moved {element_edge})" if title else "partial dual"
    for el in circuit._elements:
        if el is X or el is gyr:
            continue
        _readd_element(D, el)

    # the dual element on the far port, value carried by the conservation law
    name, t, h = X._edge.name, far.tail, far.head
    if isinstance(X, Capacitor):
        D.add_inductor(name, t, h, L=sp.simplify(X.C / G**2))
    elif isinstance(X, Inductor):
        D.add_capacitor(name, t, h, C=sp.simplify(G**2 * X.L))
    elif isinstance(X, JosephsonJunction):
        D.add_qps(name, t, h, ES=X.EJ, winding=sp.simplify(G * X.winding))
    elif isinstance(X, QuantumPhaseSlip):
        D.add_josephson(name, t, h, EJ=X.ES, winding=sp.simplify(G * X.winding))
    else:  # pragma: no cover - defensive
        raise TypeError(f"cannot move element {X!r}")

    D.ground = circuit.ground if circuit.ground in D.vertices else None
    return D

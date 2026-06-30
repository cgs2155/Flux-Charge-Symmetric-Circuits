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


def _add_dual_oneport(dest: Circuit, X, name, t, h, G):
    """Add the dual of one-port *X* between nodes *t*, *h* of *dest*, with the
    gyration ratio carried by the conservation law ``phi_near = q_far / G``:

    * capacitor ``C``  ->  inductor ``L = C / G**2``
    * inductor ``L``   ->  capacitor ``C = G**2 * L``
    * Josephson ``E_J`` -> quantum phase slip ``E_S`` (cosine argument / G)
    * quantum phase slip ``E_S`` -> Josephson ``E_J`` (cosine argument / G)
    """
    if isinstance(X, Capacitor):
        dest.add_inductor(name, t, h, L=sp.simplify(X.C / G**2))
    elif isinstance(X, Inductor):
        dest.add_capacitor(name, t, h, C=sp.simplify(G**2 * X.L))
    elif isinstance(X, JosephsonJunction):
        dest.add_qps(name, t, h, ES=X.EJ, winding=sp.simplify(G * X.winding))
    elif isinstance(X, QuantumPhaseSlip):
        dest.add_josephson(name, t, h, EJ=X.ES, winding=sp.simplify(G * X.winding))
    else:  # pragma: no cover - defensive
        raise TypeError(f"cannot move element {X!r}")


def move_across_gyrator(circuit: Circuit, element_edges) -> Circuit:
    """Move a reciprocal block across the gyrator terminating its port.

    Implements the manuscript's **partial-dual move** (Sec. "Partial Dual
    Transformations"): a reciprocal sub-network that *terminates* one half-edge
    of a gyrator is relocated to the gyrator's other port as its **dual**, with
    the gyration ratio carried by the conservation law ``phi_near = q_far / G``.

    ``element_edges`` is one edge name (single one-port) or a list of edge names
    forming a **parallel** block (sharing the two port nodes) on one gyrator
    port.  Duality swaps parallel for series, so a parallel block
    ``X_1 || X_2 || ...`` becomes the **series chain** of duals
    ``dual(X_1) - dual(X_2) - ...`` on the far port.  Per element:

    * capacitor ``C``  ->  inductor ``L = C / G**2``
    * inductor ``L``   ->  capacitor ``C = G**2 * L``
    * Josephson ``E_J`` <-> quantum phase slip ``E_S`` (cosine argument / G)

    Two cases, by how much of the port you move:

    * **Whole port** (every non-gyrative element on it) -- the source port
      empties, so the gyrator is removed by the open/closed-terminated deletion
      rule and the dual chain is strung between the far port's two nodes.
    * **Strict subset** -- the port keeps elements, so the **gyrator is
      retained**: its far half-edge is split (``far.tail -> m_0 -> ... ->
      far.head``) and the moved duals are inserted in series along that split,
      in series with the gyrator's far port.  This is the manuscript's general
      "continuous families of partial dual circuits": you can put many elements
      on both ports and slide any subset across, leaving the gyrator in place.
      A single element, or a *uniform* block (all capacitors, or all inductors),
      reduces cleanly.  A subset of *unlike* elements (capacitors and inductors
      together) leaves a retained gyrator across a mixed L/C series chain whose
      reduction is a degenerate linear cycle this reducer does not yet resolve:
      it raises ``ReductionError`` -- move a uniform block, or the whole port.

    The move is a point transformation: it **preserves the spectrum and
    well-posedness** in both cases (verified --
    e.g. a transmon ``JJ || C`` across a gyrator becomes a series ``QPS - L``
    with the same spectrum, and a well-posed input maps to a well-posed output;
    an ill-posed input -- a junction handed an effective *inductance*, say -- maps
    to an ill-posed output, as it must).

    For a **linear** element any ``G`` is fine (the ratio is in the value).  For
    a **nonlinear** element the ratio lands in the cosine argument ``cos(q / G)``
    (carried as ``winding = G``): spectrally valid for any ``G``, a standard
    element only at ``|G| = 1``, and for a compact charge representable only when
    ``1/G`` is an integer -- so ``|G| != 1`` *warns* rather than refusing.

    External biases that the move leaves well-defined are carried over: an offset
    (gate) charge is node-local and follows any node that survives with
    capacitive edges.  A bias the move would have to *dualize through* the
    gyrator -- an offset charge on the emptied island, or any loop-keyed flux
    bias against the re-inferred output -- raises ``NotImplementedError`` instead
    of being silently dropped (apply it on the result, or remove it first).
    """
    import warnings

    if isinstance(element_edges, str):
        element_edges = [element_edges]
    if not element_edges:
        raise ValueError("element_edges is empty")

    Xs = []
    for name in element_edges:
        X = next((el for el in circuit._elements
                  if getattr(el, "_edge", None) is not None
                  and el._edge.name == name), None)
        if X is None:
            raise ValueError(f"no one-port element on edge {name!r}")
        if isinstance(X, Gyrator):
            raise ValueError(f"edge {name!r} is a gyrator, not a one-port")
        Xs.append(X)

    ports = {frozenset((X._edge.tail, X._edge.head)) for X in Xs}
    if len(ports) != 1:
        raise ValueError(
            "all moved elements must share the same two nodes (a parallel block "
            f"on one gyrator port); got ports {ports}")
    port = ports.pop()

    # find the gyrator with a half-edge across that port (the block terminates it)
    gyr = near = far = None
    for el in circuit._elements:
        if not isinstance(el, Gyrator):
            continue
        for a, b in ((el.edge1, el.edge2), (el.edge2, el.edge1)):
            if frozenset((a.tail, a.head)) == port:
                gyr, near, far = el, a, b
    if gyr is None:
        raise ValueError(
            f"the block {element_edges} does not terminate a gyrator port "
            "(no gyrator half-edge shares its two nodes)")

    # Move *all* non-gyrative elements on the port (the port empties, so the
    # gyrator is consumed by the deletion rule) or only *some* of them (a strict
    # subset: the port keeps elements, so the gyrator is **retained** and its far
    # half-edge is split to carry the moved duals in series -- the manuscript's
    # general "continuous families of partial dual circuits").
    on_port = [el for el in circuit._elements
               if not isinstance(el, Gyrator)
               and getattr(el, "_edge", None) is not None
               and frozenset((el._edge.tail, el._edge.head)) == port]
    retain_gyrator = set(map(id, on_port)) != set(map(id, Xs))

    G = gyr.G
    if any(isinstance(X, (JosephsonJunction, QuantumPhaseSlip)) for X in Xs) \
            and sp.simplify(G**2 - 1) != 0:
        warnings.warn(
            f"moving a Josephson/phase-slip element across a gyrator with "
            f"|G| != 1 (G={G}) produces a cos(q/G) element (winding = G): "
            "spectrally valid, but a standard element only at |G| = 1, and for a "
            "compact charge representable only when 1/G is an integer (the "
            "diagonalizer raises otherwise); an extended coordinate is "
            "unrestricted.", stacklevel=2)

    D = Circuit()
    title = getattr(circuit, "title", None)
    moved = "+".join(element_edges)
    D.title = f"{title} (moved {moved})" if title else "partial dual"
    for el in circuit._elements:
        if el is gyr or id(el) in set(map(id, Xs)):
            continue
        _readd_element(D, el)

    n = len(Xs)
    if not retain_gyrator:
        # whole port moved: the gyrator is consumed (already dropped above), and
        # the parallel block becomes a series chain of duals between the far
        # port's two nodes, threading intermediate nodes for interior junctions.
        t0, hN = far.tail, far.head
        nodes = [t0] + [f"_m_{moved}_{i}" for i in range(1, n)] + [hN]
        for i, X in enumerate(Xs):
            _add_dual_oneport(D, X, X._edge.name, nodes[i], nodes[i + 1], G)
    else:
        # partial move: the gyrator stays, but its far half-edge is split off the
        # far node and the moved duals are strung **in series** along the split
        # (far.tail -> m_0 -> ... -> far.head), so the retained near elements
        # still see the gyrator while the moved block reappears, dualized, in
        # series with the gyrator's far port.  Verified spectrum-preserving.
        ms = [f"_m_{moved}_{i}" for i in range(n)]
        far_split = (far.name, far.tail, ms[0])      # far edge now ends at m_0
        keep = (near.name, near.tail, near.head)      # near edge unchanged
        if near is gyr.edge1:
            D.add_gyrator(keep, far_split, G=gyr.G)
        else:
            D.add_gyrator(far_split, keep, G=gyr.G)
        chain = ms + [far.head]                       # m_0 -> ... -> far.head
        for i, X in enumerate(Xs):
            _add_dual_oneport(D, X, X._edge.name, chain[i], chain[i + 1], G)

    D.ground = circuit.ground if circuit.ground in D.vertices else None

    # carry external biases the move leaves well-defined.  An offset (gate)
    # charge is a property of its node, applied by splitting over that node's
    # capacitive edges: if the node survives the move with capacitive edges, the
    # gate charge carries over directly.  A bias whose home is consumed by the
    # move -- an offset charge on the emptied island, or any flux bias (the
    # output re-infers its loops, so a loop-keyed external flux has no stable
    # target) -- would have to be *dualized through the gyrator*, a partial-dual
    # mapping not derived here; refuse rather than silently drop or mis-place it.
    def _has_capacitive_edge(node):
        return any(isinstance(el, (Capacitor, QuantumPhaseSlip))
                   and node in (el._edge.tail, el._edge.head)
                   for el in D._elements)

    for node, ng in circuit._offset_charge.items():
        if node in D.vertices and _has_capacitive_edge(node):
            D.set_offset_charge(node, ng)
        else:
            raise NotImplementedError(
                f"offset charge on node {node!r} sits on the part of the circuit "
                "carried across the gyrator (its capacitive anchor is gone), so "
                "the partial dual would map it to an external flux -- a "
                "transformation not yet derived. Remove the bias before moving, "
                "or apply the equivalent flux on the result.")
    if circuit._flux_bias:
        raise NotImplementedError(
            "flux-bias carry-over across move_across_gyrator is not supported: "
            "the moved circuit re-infers its loops, so a loop-keyed external flux "
            f"has no stable target (biased loops {list(circuit._flux_bias)}). "
            "Remove the flux bias before moving, or apply it on the result after "
            "inspecting its inferred loops.")
    return D

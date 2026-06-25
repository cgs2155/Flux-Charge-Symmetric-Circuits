"""
Lumped-element schematic drawing with real circuit symbols.

Where :mod:`fluxcharge.visualize` draws the *topology* (a networkx graph),
this module draws a proper **schematic**: every element is rendered with its
standard symbol (inductor coil, capacitor plates, the Josephson-junction
boxed-X, a diamond for the quantum phase slip, and a gyrator
coupling) and every wire is a straight line.

It is built on `schemdraw <https://schemdraw.readthedocs.io>`_ (matplotlib
backend, no LaTeX required).  Node positions come from a layout of the graph
reconstructed from the incidence matrix ``A``; each element is then placed as a
straight two-terminal symbol between its endpoints, and parallel branches
between the same pair of nodes are routed as offset straight stubs (so, e.g.,
an LC loop is drawn as a rectangle rather than two overlapping symbols).

Install the optional dependency with ``pip install "fluxcharge[schematic]"``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple


def _qps_element():
    """Quantum-phase-slip junction symbol (the charge-space dual of the JJ).

    Drawn as a *diamond* whose left and right vertices sit on the wire, with a
    *center line normal to the wire* (the phase slip occurs across the wire).
    Because the element is defined along its local x-axis, the diamond's
    long axis lies along the wire and the center line is drawn along local y,
    so it stays perpendicular to the wire at any edge angle once schemdraw
    rotates the element.
    """
    import schemdraw.elements as elm
    from schemdraw.segments import Segment

    class QPS(elm.Element2Term):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            dx0, dx1, dy = 0.70, 1.30, 0.30
            cx = 0.5 * (dx0 + dx1)
            # leads (meet the diamond at its on-wire vertices)
            self.segments.append(Segment([(0, 0), (dx0, 0)]))
            self.segments.append(Segment([(dx1, 0), (2.0, 0)]))
            # diamond: left and right vertices on the wire, top/bottom off it
            self.segments.append(Segment([(dx0, 0), (cx, dy),
                                          (dx1, 0), (cx, -dy), (dx0, 0)]))
            # center line, normal to the wire, kept inside the diamond
            self.segments.append(Segment([(cx, -dy), (cx, dy)], lw=2.5))
            self.anchors['start'] = (0, 0)
            self.anchors['end'] = (2.0, 0)
    return QPS


def _half_gyrator_element():
    """The manuscript's *half gyrator* symbol: a straight wire carrying a
    thicker semicircular crescent.  The crescent's diameter lies along the
    wire, so it bulges normal to the line at any angle (schemdraw rotates it
    with the edge).  Two half gyrators, one per gyrator edge, form the gyrator.
    """
    import schemdraw.elements as elm
    from schemdraw.segments import Segment, SegmentArc

    class HalfGyrator(elm.Element2Term):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            r = 0.45
            self.segments.append(Segment([(0, 0), (2.0, 0)]))
            self.segments.append(SegmentArc((1.0, 0), 2 * r, 2 * r,
                                            theta1=0, theta2=180, lw=3))
            self.anchors['start'] = (0, 0)
            self.anchors['end'] = (2.0, 0)
    return HalfGyrator


_ELEMENT_FACTORY = {
    "Capacitor":         ("Capacitor",  "C"),
    "Inductor":          ("Inductor2",  "L"),
    "JosephsonJunction": ("Josephson",  "E_J"),
}


def _outer_loop_name(circuit, outer_loop):
    """Pick the face to use as the outer boundary.

    An explicit choice wins.  Otherwise the outer face is selected by a
    deterministic, declaration-order-independent rule so that topologically
    identical circuits (e.g. a circuit and its double dual, which differ only in
    the order their loops were declared) always draw the same way: prefer the
    largest face (most boundary edges), then the face carrying the fewest
    gyrative edges (gyrators read as interior couplings, not outer boundary),
    breaking any remaining tie on the sorted edge-name list.
    """
    if outer_loop is not None and outer_loop in circuit._loops:
        return outer_loop
    if not circuit._loops:
        return None
    from .elements import GYRATIVE

    def key(loop):
        entries = circuit._loops[loop]
        edges = [e for _, e in entries]
        n_gyr = sum(1 for e in edges if circuit._edges[e].edge_class == GYRATIVE)
        return (-len(entries), n_gyr, tuple(sorted(edges)))

    return min(circuit._loops, key=key)


def _cycle_vertices(circuit, loopname):
    """Ordered vertices around a loop, walking its signed edges as a closed
    boundary (a face of the planar embedding)."""
    steps = []
    for s, e in circuit._loops[loopname]:
        edge = circuit._edges[e]
        steps.append((edge.tail, edge.head) if s > 0 else (edge.head, edge.tail))
    succ = {}
    for a, b in steps:
        succ.setdefault(a, []).append(b)
    start = steps[0][0]
    seq, seen, cur = [start], {start}, succ[start][0]
    while cur not in seen:
        seq.append(cur)
        seen.add(cur)
        nxt = None
        for w in succ.get(cur, []):
            if w not in seen:
                nxt = w
                break
        if nxt is None:
            break
        cur = nxt
    return seq


def _planar_layout(circuit, outer_loop, scale):
    """A crossing-free straight-line layout from the planar face structure.

    The outer face (from ``B``) is placed on a regular convex polygon; interior
    vertices are solved by the barycentric (Tutte) method, each sitting at the
    average of its neighbours.  For a 3-connected planar graph this is
    guaranteed crossing-free with convex faces.
    """
    import numpy as np

    verts = circuit.vertices
    outer = _outer_loop_name(circuit, outer_loop)
    cyc = [v for v in _cycle_vertices(circuit, outer) if v in verts] if outer else []
    # dedupe preserving order
    seen = set()
    cyc = [v for v in cyc if not (v in seen or seen.add(v))]
    if len(cyc) < 2:
        return None

    pos = {}
    n = len(cyc)
    if n == 2:
        pos[cyc[0]] = np.array([-scale, 0.0])
        pos[cyc[1]] = np.array([scale, 0.0])
    else:
        # regular polygon oriented to rest on a flat *top* edge, so the first
        # boundary edge reads horizontally across the top and the figure is
        # upright (matching how circuits are usually drawn)
        for i, v in enumerate(cyc):
            ang = np.pi / 2 + np.pi / n - 2 * np.pi * i / n
            pos[v] = np.array([np.cos(ang), np.sin(ang)]) * scale

    interior = [v for v in verts if v not in set(cyc)]
    if interior:
        import networkx as nx
        U = circuit.to_networkx().to_undirected()
        idx = {v: i for i, v in enumerate(interior)}
        M = np.zeros((len(interior), len(interior)))
        rhs = np.zeros((len(interior), 2))
        for v in interior:
            i = idx[v]
            nbrs = list(U.neighbors(v))
            M[i, i] = len(nbrs) or 1
            for w in nbrs:
                if w in idx:
                    M[i, idx[w]] -= 1
                else:
                    rhs[i] += pos[w]
        sol = np.linalg.solve(M, rhs)
        for v in interior:
            pos[v] = sol[idx[v]]
    return pos


def _layout(circuit, layout: str, scale: float):
    """Node positions, scaled to schemdraw units."""
    import numpy as np
    import networkx as nx

    G = circuit.to_networkx()

    if layout in ("auto", "planar"):
        pos = _planar_layout(circuit, None, scale)
        if pos is not None:
            return {k: np.asarray(v, dtype=float) for k, v in pos.items()}, G

    simple = nx.Graph()
    simple.add_nodes_from(G.nodes())
    simple.add_edges_from((u, v) for u, v, _ in G.edges(keys=True) if u != v)
    if layout == "circular":
        pos = nx.circular_layout(simple, scale=scale)
    else:
        pos = nx.spring_layout(simple, seed=7, scale=scale)
    return {k: np.asarray(v, dtype=float) for k, v in pos.items()}, G


def draw_schematic(circuit, file: Optional[str] = None, layout: str = "auto",
                   unit: float = 3.0, scale: float = 3.2, show_values: bool = True,
                   positions: Optional[Dict[str, Tuple[float, float]]] = None,
                   outer_loop: Optional[str] = None):
    """Draw *circuit* as a lumped-element schematic.

    Parameters
    ----------
    file : str, optional
        If given, the schematic is written here (``.png``, ``.svg``, ``.pdf``).
    layout : {"auto", "circular", "planar", "spring"}
        Node placement when *positions* is not supplied.  ``"auto"`` uses a
        circular layout for small circuits.
    unit, scale : float
        ``unit`` is the schemdraw element body length; ``scale`` sets the
        node-to-node spacing.  Keep ``scale`` >= ``unit`` so symbols are not
        cramped.
    show_values : bool
        Label each element with its symbolic parameter.
    positions : dict, optional
        Explicit ``{node: (x, y)}`` coordinates (in schemdraw units).  Supplying
        these -- for example from a grid or an interactive editor -- gives the
        cleanest, orthogonal schematics, since automatic layout cannot in
        general route an arbitrary circuit without diagonal wires.

    Returns
    -------
    schemdraw.Drawing
    """
    import numpy as np
    import schemdraw
    import schemdraw.elements as elm

    # The crossing-free planar layout reads the circuit's faces (loops).  When
    # none were declared, infer them from the graph -- exactly as hamiltonian()
    # does -- so a circuit built without explicit loops still draws planar
    # instead of falling back to the tangled spring layout.
    if not circuit._loops:
        try:
            circuit.infer_loops()
        except Exception:
            pass

    # a circuit may carry a preferred layout (e.g. the library 0-pi ships its
    # triangle-with-centre positions); use it when the caller gives none
    if positions is None:
        positions = getattr(circuit, "_positions", None)

    if positions is not None:
        pos = {k: np.asarray(v, dtype=float) for k, v in positions.items()}
        G = circuit.to_networkx()
    else:
        if layout in ("auto", "planar"):
            p = _planar_layout(circuit, outer_loop, scale)
            if p is not None:
                pos = {k: np.asarray(v, dtype=float) for k, v in p.items()}
                G = circuit.to_networkx()
            else:
                pos, G = _layout(circuit, layout, scale)
        else:
            pos, G = _layout(circuit, layout, scale)
    QPS = _qps_element()
    HalfGyrator = _half_gyrator_element()

    # group parallel edges (same unordered node pair)
    groups: Dict[frozenset, list] = {}
    for u, v, k in G.edges(keys=True):
        groups.setdefault(frozenset((u, v)), []).append((u, v, k))

    # which edges lie on the outer face -> those are routed around the outside,
    # keeping their interior siblings as straight chords (planar, no crossings)
    outer_name = _outer_loop_name(circuit, outer_loop)
    outer_edges = ({e for _, e in circuit._loops[outer_name]}
                   if outer_name is not None else set())
    centroid = np.mean(np.array(list(pos.values())), axis=0)
    n_nodes = len(pos)

    # gyrator bookkeeping: each half-gyrator's partner and the centre of every
    # edge, so a crescent can be oriented to face its partner (the grey coupling
    # line) while remaining normal to its own wire.
    gyr_pairs = _gyrator_pairs(circuit)
    gyr_partner: Dict[str, str] = {}
    for (n1, n2, _ratio) in gyr_pairs:
        gyr_partner[n1] = n2
        gyr_partner[n2] = n1
    edge_endpoints = {G.get_edge_data(u, v, k)["name"]: (u, v)
                      for u, v, k in G.edges(keys=True)}
    edge_centre = {nm: 0.5 * (pos[u] + pos[v]) for nm, (u, v) in edge_endpoints.items()}

    gyr_port_centres: Dict[str, np.ndarray] = {}

    d = schemdraw.Drawing(show=False)
    d.config(unit=unit, fontsize=12)

    def _place(etype, A, B, param, name):
        if etype == "QuantumPhaseSlip":
            e = QPS().at(tuple(A)).to(tuple(B))
            sym = "E_S"
        elif etype == "JosephsonJunction":
            e = elm.Josephson(box=True).at(tuple(A)).to(tuple(B))
            sym = None
        elif etype == "Gyrator":
            e = HalfGyrator().at(tuple(A)).to(tuple(B))
            sym = None
        else:
            cls, _ = _ELEMENT_FACTORY.get(etype, ("ResistorIEC", "?"))
            e = getattr(elm, cls)().at(tuple(A)).to(tuple(B))
            sym = None
        if show_values and param is not None and etype != "Gyrator":
            e = e.label(f"${_latexify(param)}$", fontsize=12)
        e = e.label(name, loc="bottom", fontsize=9, color="#888")
        d.add(e)
        return e

    for pair, elist in groups.items():
        m = len(elist)
        na, nb = sorted(pair)
        base = pos[nb] - pos[na]
        L = float(np.linalg.norm(base)) or scale
        uhat = (base / L) if L else np.array([1.0, 0.0])
        base_nhat = np.array([-uhat[1], uhat[0]])

        # outward perpendicular direction for this edge (away from centroid),
        # so parallel siblings bow to the outside of the layout, not inward
        mid = 0.5 * (pos[na] + pos[nb])
        outward = mid - centroid
        out_sign = 1.0 if float(np.dot(outward, base_nhat)) >= 0 else -1.0

        # decide a perpendicular offset and a drawing style per element:
        #   "chord" -- drawn full length along the edge (the lone/main element)
        #   "fan"   -- isolated bundle (e.g. an LC loop): symmetric arcs
        #   "rung"  -- a parallel sibling: a SHORT element near the midpoint,
        #              offset outward and tapped off the chord, so two elements
        #              on one node pair read as a tight ladder rather than a big
        #              full-length parallelogram swinging across the figure
        placements = []   # (edge_key, off, style)
        if m == 1:
            placements.append((elist[0], 0.0, "chord"))
        elif n_nodes <= 2:
            sib = min(0.42 * L, 0.55 * unit)
            for e, off in zip(elist, np.linspace(-1.0, 1.0, m) * sib):
                placements.append((e, float(off), "fan"))
        else:
            rung = min(0.5 * unit, 0.3 * L)
            placements.append((elist[0], 0.0, "chord"))
            for i, e in enumerate(elist[1:], start=1):
                placements.append((e, out_sign * rung * i, "rung"))

        for (u, v, k), off, style in placements:
            data = G.get_edge_data(u, v, k)
            etype = data["element_type"]
            P, Q = pos[u], pos[v]
            if np.linalg.norm(Q - P) == 0:
                continue
            param = data.get("parameter")

            if style == "rung":
                half = min(0.32 * L, 0.7 * unit)
                Sa, Sb = mid - uhat * half, mid + uhat * half
                A = Sa + base_nhat * off
                B = Sb + base_nhat * off
            else:
                A = P + base_nhat * off
                B = Q + base_nhat * off

            # orient the half-gyrator crescent so it bulges toward its partner
            # (the grey coupling line).  The crescent bulges 90 deg CCW from the
            # draw direction A->B, so swap the endpoints when that points away
            # from the partner -- this keeps it normal to the wire either way.
            if etype == "Gyrator":
                nm = data["name"]
                partner = gyr_partner.get(nm)
                if partner in edge_centre:
                    self_c = 0.5 * (A + B)
                    d_bulge = edge_centre[partner] - self_c
                    t = B - A
                    n_plus = np.array([-t[1], t[0]])
                    if float(np.dot(n_plus, d_bulge)) < 0:
                        A, B = B, A

            if style == "rung":
                d += elm.Line().at(tuple(Sa)).to(tuple(A))
                _place(etype, A, B, param, data["name"])
                d += elm.Line().at(tuple(B)).to(tuple(Sb))
            else:
                if off != 0:
                    d += elm.Line().at(tuple(P)).to(tuple(A))
                _place(etype, A, B, param, data["name"])
                if off != 0:
                    d += elm.Line().at(tuple(B)).to(tuple(Q))
            if etype == "Gyrator":
                gyr_port_centres[data["name"]] = 0.5 * (A + B)
            d += elm.Dot().at(tuple(P))
            d += elm.Dot().at(tuple(Q))

    # gyrator coupling: faint dotted link between the two half-gyrator ports
    for (n1, n2, ratio) in gyr_pairs:
        if n1 in gyr_port_centres and n2 in gyr_port_centres:
            c1, c2 = gyr_port_centres[n1], gyr_port_centres[n2]
            d += elm.Line().at(tuple(c1)).to(tuple(c2)).linestyle(":").color("#9a9a9a")
            mid = 0.5 * (c1 + c2)
            d += elm.Label().at(tuple(mid)).label(f"${_latexify(ratio)}$",
                                                  fontsize=10, color="#5b5b5b")

    # node name labels
    for name, p in pos.items():
        d += elm.Label().at((p[0], p[1] + 0.35)).label(name, fontsize=11)

    if file is not None:
        d.save(file)
    return d


def _latexify(sym) -> str:
    import sympy as sp
    return sp.latex(sym)


def _gyrator_pairs(circuit):
    """Return list of (edge1_name, edge2_name, ratio) for each gyrator."""
    out = []
    seen = set()
    for ename, edge in circuit._edges.items():
        el = edge.element
        if type(el).__name__ == "Gyrator" and id(el) not in seen:
            seen.add(id(el))
            out.append((el.edge1.name, el.edge2.name, el.G))
    return out

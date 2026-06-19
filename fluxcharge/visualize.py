"""
Drawing a circuit as a graph, reconstructed from the boundary matrices.

The topology of the circuit is carried entirely by the two boundary maps built
by :class:`~fluxcharge.circuit.Circuit`:

* the incidence matrix ``A`` (``|E| x |V|``) with ``A[e, head] = +1`` and
  ``A[e, tail] = -1`` -- this fixes which two vertices each edge joins and the
  edge's orientation;
* the loop/orientation matrix ``B`` (``|F| x |E|``) whose signed rows list the
  edges bounding each face.

:func:`circuit_to_networkx` rebuilds a :class:`networkx.MultiDiGraph` from ``A``
alone (so it is a faithful picture of the topology the formalism actually
uses), tags the faces from ``B``, and decorates each edge with its element
class and type for styling.  :func:`draw_circuit` renders it with matplotlib.

These are intended both for inspecting a circuit and as the data backbone for a
future interactive (drag-and-drop) builder: the returned graph is a plain
networkx object whose nodes are circuit nodes and whose edges are elements.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Per-element-type drawing style: (colour, short label, linestyle).
_ELEMENT_STYLE: Dict[str, Tuple[str, str, str]] = {
    "Inductor":          ("#1f5fbf", "L",   "solid"),
    "Capacitor":         ("#d1495b", "C",   "solid"),
    "JosephsonJunction": ("#7b3fa0", "JJ",  "solid"),
    "QuantumPhaseSlip":  ("#1b998b", "QPS", "solid"),
    "Gyrator":           ("#5b5b5b", "G",   "dashed"),
}
_DEFAULT_STYLE = ("#333333", "?", "solid")


def _edge_endpoints_from_A(A, ei: int, vertices: List[str]) -> Tuple[str, str]:
    """Recover ``(tail, head)`` of edge row *ei* from the incidence matrix."""
    tail = head = None
    for vi, v in enumerate(vertices):
        val = A[ei, vi]
        if val == 1:
            head = v
        elif val == -1:
            tail = v
    return tail, head


def circuit_to_networkx(circuit):
    """Reconstruct a :class:`networkx.MultiDiGraph` from ``A`` and ``B``.

    Nodes are the circuit vertices.  Each edge (``tail -> head``, keyed by the
    element-edge name) carries:

    ``edge_class``    one of ``"C"``, ``"I"``, ``"G"``;
    ``element_type``  e.g. ``"Capacitor"``, ``"JosephsonJunction"``;
    ``parameter``     the element's symbolic parameter (``C``, ``L``, ``E_J``,
                      ``E_S`` or gyration ratio ``G``), if any.

    The graph attribute ``G.graph["loops"]`` maps each loop name to the signed
    edge list read back from ``B``.
    """
    import networkx as nx

    A = circuit.incidence_matrix()
    B = circuit.orientation_matrix()
    vertices = circuit.vertices
    edges = circuit.edges
    loops = circuit.loops

    G = nx.MultiDiGraph()
    for v in vertices:
        G.add_node(v)

    for ei, ename in enumerate(edges):
        tail, head = _edge_endpoints_from_A(A, ei, vertices)
        edge = circuit._edges[ename]
        element = edge.element
        etype = type(element).__name__ if element is not None else "Unknown"
        if etype == "Gyrator":
            param = element.G
        else:
            param = next(iter(element.parameters), None) if element is not None else None
        G.add_edge(
            tail, head, key=ename,
            name=ename,
            edge_class=edge.edge_class,
            element_type=etype,
            parameter=param,
        )

    eidx = {e: i for i, e in enumerate(edges)}
    face_map = {}
    for li, lname in enumerate(loops):
        members = []
        for ename in edges:
            s = B[li, eidx[ename]]
            if s != 0:
                members.append((int(s), ename))
        face_map[lname] = members
    G.graph["loops"] = face_map
    return G


def _layout(G, layout: str):
    import networkx as nx

    simple = nx.Graph()
    simple.add_nodes_from(G.nodes())
    simple.add_edges_from((u, v) for u, v, _ in G.edges(keys=True) if u != v)

    if layout == "auto":
        # circular spreads small circuits cleanly (a triangle for 3 nodes, two
        # well-separated points for 2); spring scales better past that.
        if simple.number_of_nodes() <= 6:
            return nx.circular_layout(simple, scale=1.0)
        return nx.spring_layout(simple, seed=7, k=1.2)
    if layout == "planar":
        ok, embedding = nx.check_planarity(simple)
        if ok:
            try:
                return nx.planar_layout(embedding, scale=1.0)
            except Exception:
                pass
        return nx.spring_layout(simple, seed=7, k=1.2)
    if layout == "circular":
        return nx.circular_layout(simple, scale=1.0)
    if layout == "shell":
        return nx.shell_layout(simple)
    return nx.spring_layout(simple, seed=7, k=1.2)


def draw_circuit(circuit, ax=None, layout: str = "auto", show_loops: bool = True,
                 title: Optional[str] = None, node_size: int = 2000):
    """Draw *circuit* as a styled graph and return the matplotlib ``Axes``.

    Parameters
    ----------
    layout : {"auto", "planar", "spring", "circular", "shell"}
        Node placement.  ``"auto"`` uses a planar embedding when the circuit is
        planar (the usual case) and falls back to a spring layout otherwise.
    show_loops : bool
        Annotate each face (loop) from ``B`` at the centroid of its edges.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    G = circuit_to_networkx(circuit)
    pos = {k: np.asarray(v, dtype=float) for k, v in _layout(G, layout).items()}

    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 6.5))

    # nodes
    for v, (x, y) in pos.items():
        ax.scatter([x], [y], s=node_size, facecolor="white",
                   edgecolor="#222", linewidths=1.6, zorder=3)
        ax.text(x, y, v, ha="center", va="center", fontsize=11,
                fontweight="bold", zorder=4)

    # group parallel edges (same unordered vertex pair) to fan them out
    groups: Dict[frozenset, List[Tuple[str, str, str]]] = {}
    for u, v, k in G.edges(keys=True):
        groups.setdefault(frozenset((u, v)), []).append((u, v, k))

    edge_mid: Dict[str, np.ndarray] = {}
    present_types = set()

    for pair, elist in groups.items():
        n = len(elist)
        # geometric offsets: 0 for a lone edge, fanned for parallels.  These are
        # assigned relative to a *canonical* orientation of the vertex pair so
        # that antiparallel edges (tail/head swapped) fan to opposite sides
        # rather than overlapping.
        if n == 1:
            offsets = [0.0]
        else:
            spread = 0.5
            offsets = list(np.linspace(-spread, spread, n))

        # canonical pair direction (by node label) used only to decide which
        # edges run "forward" so antiparallel arcs fan to opposite sides.
        u0, v0 = elist[0][0], elist[0][1]
        a, b = (u0, v0) if u0 == v0 else tuple(sorted((u0, v0)))

        for (u, v, k), g in zip(elist, offsets):
            data = G.get_edge_data(u, v, k)
            etype = data["element_type"]
            colour, short, ls = _ELEMENT_STYLE.get(etype, _DEFAULT_STYLE)
            present_types.add(etype)
            p_u, p_v = pos[u], pos[v]

            if u == v:  # self-loop
                patch = FancyArrowPatch(
                    p_u + np.array([0.02, 0.0]), p_v - np.array([0.02, 0.0]),
                    connectionstyle="arc3,rad=0.9", arrowstyle="-|>",
                    mutation_scale=16, lw=2.0, color=colour, linestyle=ls,
                    shrinkA=16, shrinkB=16, zorder=2)
                mid = p_u + np.array([0.0, 0.28])
            else:
                # arc3 rad is relative to the *directed* chord, so flip it when
                # this edge runs opposite to the canonical (a -> b) direction,
                # keeping the geometric bulge on the side picked by ``g``.
                forward = (u == a and v == b)
                rad = g if forward else -g
                patch = FancyArrowPatch(
                    p_u, p_v, connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-|>", mutation_scale=16, lw=2.0,
                    color=colour, linestyle=ls, shrinkA=22, shrinkB=22, zorder=2)
                # label at this edge's own arc apex: rotate the chord +90 deg and
                # offset by rad * |chord| / 2 (matplotlib's arc3 convention).
                d = p_v - p_u
                chord_len = np.linalg.norm(d)
                n_hat = np.array([-d[1], d[0]])
                if chord_len > 0:
                    n_hat = n_hat / chord_len
                mid = 0.5 * (p_u + p_v) - n_hat * rad * 0.5 * chord_len
            ax.add_patch(patch)

            param = data.get("parameter")
            label = data["name"] if param is None else f"{data['name']}  {short}={param}"
            edge_mid[data["name"]] = mid
            ax.text(mid[0], mid[1], label, ha="center", va="center", fontsize=8.5,
                    color=colour, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=colour, lw=0.8, alpha=0.92))

    # faces / loops from B
    if show_loops:
        for lname, members in G.graph["loops"].items():
            pts = [edge_mid[en] for _, en in members if en in edge_mid]
            if not pts:
                continue
            c = np.mean(pts, axis=0)
            ax.text(c[0], c[1], lname, ha="center", va="center", fontsize=9,
                    style="italic", color="#444", zorder=1,
                    bbox=dict(boxstyle="circle,pad=0.25", fc="#f3f3f3",
                              ec="#bbb", lw=0.8, alpha=0.85))

    # legend
    from matplotlib.lines import Line2D
    handles = []
    for etype in sorted(present_types):
        colour, short, ls = _ELEMENT_STYLE.get(etype, _DEFAULT_STYLE)
        handles.append(Line2D([0], [0], color=colour, lw=2.2, linestyle=ls,
                              label=f"{etype} ({short})"))
    if show_loops and G.graph["loops"]:
        handles.append(Line2D([0], [0], marker="o", color="none",
                              markerfacecolor="#f3f3f3", markeredgecolor="#bbb",
                              markersize=10, label="loop / face (from B)"))
    if handles:
        ax.legend(handles=handles, loc="upper left", fontsize=8.5,
                  frameon=True, framealpha=0.9, bbox_to_anchor=(1.0, 1.0))

    # FancyArrowPatch arcs do NOT update the data limits, so set them by hand
    # from the node positions plus the edge label points (which sit near the
    # arc apexes); otherwise collinear layouts clip the curved edges.
    import numpy as np
    pts = list(pos.values()) + list(edge_mid.values())
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    pad = 0.22 * span
    cx, cy = 0.5 * (max(xs) + min(xs)), 0.5 * (max(ys) + min(ys))
    half = 0.5 * span + pad
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)

    ax.set_title(title or "circuit graph (nodes from A, faces from B)",
                 fontsize=12, pad=16)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax

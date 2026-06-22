"""
Automatic loop inference: derive a circuit's loops from its graph so the user
need not declare faces by hand.

Two routines, picked automatically:

* **Planar** (the usual case) -- find a planar embedding (via ``networkx``) and
  trace its *faces*.  Faces are what the flux-charge symmetric formalism needs:
  they carry the vertices<->faces duality, so the full machinery
  (``hamiltonian``, ``dual``, ``schematic``) works.

* **Non-planar fallback** -- a non-planar graph has no faces, but the
  Hamiltonian only needs a basis of the cycle space.  We return a tree-based
  fundamental cycle basis (``E - V + 1`` loops).  The reduction works; ``dual``
  and ``schematic`` do not (there is no planar dual), and a warning is emitted.

Parallel edges (e.g. a junction shunted by a capacitor and an inductor) are
handled by subdividing every edge with a midpoint node before embedding, then
mapping faces back to the original directed edges.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple


def _edge_ends(circuit):
    return {e: (circuit._edges[e].tail, circuit._edges[e].head) for e in circuit.edges}


def _sign_token(ename, a, b, ends):
    """``+ename`` if traversing ``a->b`` matches the edge's tail->head, else ``-``."""
    tail, head = ends[ename]
    return ("+" if (tail, head) == (a, b) else "-") + ename


def _planar_faces(circuit) -> Optional[List[Tuple[str, List[str]]]]:
    """All faces of a planar embedding as signed edge lists, or None if the graph
    is non-planar."""
    import networkx as nx

    ends = _edge_ends(circuit)
    # subdivide each edge e (u->v) into u - mid_e - v so parallel edges become
    # distinct paths and the graph is simple (required by check_planarity)
    G = nx.Graph()
    mids = {}
    for e, (u, v) in ends.items():
        m = ("mid", e)
        mids[m] = e
        G.add_edge(u, m)
        G.add_edge(m, v)
    if G.number_of_nodes() == 0:
        return []

    ok, emb = nx.check_planarity(G)
    if not ok:
        return None

    faces, seen = [], set()
    for u, v in list(emb.edges()):
        for x, y in ((u, v), (v, u)):
            if (x, y) in seen:
                continue
            nodes = emb.traverse_face(x, y, mark_half_edges=seen)
            faces.append(nodes)

    loops = []
    for k, nodes in enumerate(faces):
        n = len(nodes)
        tokens = []
        for i, node in enumerate(nodes):
            if node in mids:                      # a subdivision midpoint = one edge
                prev_node = nodes[(i - 1) % n]
                next_node = nodes[(i + 1) % n]
                tokens.append(_sign_token(mids[node], prev_node, next_node, ends))
        if tokens:
            loops.append((f"f{k + 1}", tokens))
    return loops


def _fundamental_cycles(circuit) -> List[Tuple[str, List[str]]]:
    """Tree-based fundamental cycle basis (``E - V + 1`` loops), multigraph-aware.
    Works for any graph but yields cycles, not planar faces."""
    import networkx as nx

    ends = _edge_ends(circuit)
    tree = nx.Graph()                  # spanning forest; edge attr 'key' = edge name
    tree.add_nodes_from(circuit.vertices)
    in_tree = set()
    seen = set()
    # BFS spanning forest, recording which edge name connects each tree edge
    adj = {v: [] for v in circuit.vertices}
    for e, (u, v) in ends.items():
        adj[u].append((v, e))
        adj[v].append((u, e))
    for root in circuit.vertices:
        if root in seen:
            continue
        seen.add(root)
        stack = [root]
        while stack:
            a = stack.pop()
            for b, e in adj[a]:
                if b not in seen:
                    seen.add(b)
                    in_tree.add(e)
                    tree.add_edge(a, b, key=e)
                    stack.append(b)

    loops = []
    k = 0
    for e, (u, v) in ends.items():
        if e in in_tree:
            continue
        try:
            path = nx.shortest_path(tree, v, u)      # v ... back to u, in the tree
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        tokens = [_sign_token(e, u, v, ends)]        # the chord, traversed u->v
        for a, b in zip(path, path[1:]):
            tokens.append(_sign_token(tree[a][b]["key"], a, b, ends))
        k += 1
        loops.append((f"f{k}", tokens))
    return loops


def infer_loops(circuit) -> Tuple[List[Tuple[str, List[str]]], bool]:
    """Infer loops for *circuit* from its graph.

    Returns ``(loops, planar)`` where *loops* is a list of
    ``(name, [signed-edge tokens])`` ready for :meth:`Circuit.add_loop`, and
    *planar* says whether they are planar faces (full functionality) or a
    non-planar cycle basis (Hamiltonian only).
    """
    faces = _planar_faces(circuit)
    if faces is not None:
        return faces, True
    warnings.warn(
        "circuit is non-planar: inferred a cycle basis (the Hamiltonian is fine, "
        "but dual() and schematic() need a planar embedding and will not work).",
        stacklevel=2)
    return _fundamental_cycles(circuit), False

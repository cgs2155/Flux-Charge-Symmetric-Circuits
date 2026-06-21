"""
A small text "netlist" format for entering a circuit.

The format is line-based and human-writable.  Blank lines and anything after a
``#`` are ignored.  Each remaining line is one of:

Two-terminal elements -- ``TYPE  name  node1  node2  [value]`` ::

    C    e2  v2 v3  C        # capacitor   (value -> capacitance symbol)
    L    e1  v1 v2  L        # inductor    (value -> inductance symbol)
    J    e1  v1 v2  E_J      # Josephson junction (value -> Josephson energy)
    QPS  e1  v1 v2  E_S      # quantum phase slip  (value -> phase-slip energy)

The optional *value* is a symbol name (e.g. ``C``, ``E_J``) or a number.  If two
elements share a value name they share that parameter (as the manuscript's two
capacitors both use ``C``).  If omitted, a unique symbol is generated.

Gyrator -- ``gyrator  edge1 n1a n1b   edge2 n2a n2b   [ratio]`` ::

    gyrator  e4 v1 v3   e5 v2 v3   G

Loops / faces of the planar circuit -- ``loop  name  signed-edges...`` ::

    loop  f4  -e1 -e2 -e3

Gauge (optional) -- a ground node and the open (outer) loop ::

    ground v1
    open   f4

An optional ``title  <text>`` line names the circuit.

Example (the manuscript circulator)::

    title Circulator
    J    e1  v1 v2  E_J
    C    e2  v2 v3  C
    C    e3  v3 v1  C
    gyrator  e4 v1 v3   e5 v2 v3   G
    loop f1  +e3 +e4
    loop f2  +e1 -e4 +e5
    loop f3  +e2 -e5
    loop f4  -e1 -e2 -e3
    ground v1
    open   f4
"""

from __future__ import annotations

import os
from typing import List, Optional

from .circuit import Circuit

_ELEMENT_ALIASES = {
    "C": "capacitor",
    "CAP": "capacitor",
    "L": "inductor",
    "IND": "inductor",
    "J": "josephson",
    "JJ": "josephson",
    "QPS": "qps",
    "S": "qps",
}


class NetlistError(ValueError):
    """Raised on a malformed netlist line, with the line number."""


def _strip(line: str) -> str:
    return line.split("#", 1)[0].strip()


def parse_netlist(text: str) -> Circuit:
    """Parse netlist *text* into a :class:`~fluxcharge.circuit.Circuit`.

    The returned circuit carries ``ground``, ``open_loops`` and ``title``
    attributes from the directives (used as defaults by the CLI / helpers).
    """
    ckt = Circuit()
    ckt.title = None
    ckt.ground = None
    ckt.open_loops: List[str] = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _strip(raw)
        if not line:
            continue
        tok = line.split()
        head = tok[0]
        key = head.upper()

        try:
            if key in _ELEMENT_ALIASES:
                kind = _ELEMENT_ALIASES[key]
                if len(tok) < 4:
                    raise NetlistError(
                        f"line {lineno}: '{head}' needs 'name node1 node2 [value]'")
                name, n1, n2 = tok[1], tok[2], tok[3]
                value = tok[4] if len(tok) >= 5 else None
                if kind == "capacitor":
                    ckt.add_capacitor(name, n1, n2, C=value or f"C_{name}")
                elif kind == "inductor":
                    ckt.add_inductor(name, n1, n2, L=value or f"L_{name}")
                elif kind == "josephson":
                    ckt.add_josephson(name, n1, n2, EJ=value or f"E_J_{name}")
                elif kind == "qps":
                    ckt.add_qps(name, n1, n2, ES=value or f"E_S_{name}")

            elif head.lower() == "gyrator":
                if len(tok) < 7:
                    raise NetlistError(
                        f"line {lineno}: 'gyrator' needs "
                        "'edge1 n1a n1b edge2 n2a n2b [ratio]'")
                e1, a1, b1, e2, a2, b2 = tok[1:7]
                ratio = tok[7] if len(tok) >= 8 else "G"
                ckt.add_gyrator((e1, a1, b1), (e2, a2, b2), G=ratio)

            elif head.lower() == "loop":
                if len(tok) < 3:
                    raise NetlistError(
                        f"line {lineno}: 'loop' needs 'name signed-edges...'")
                name = tok[1]
                edges = [e.lstrip(":") for e in tok[2:] if e != ":"]
                # allow an optional ':' after the loop name
                name = name.rstrip(":")
                ckt.add_loop(name, edges)

            elif head.lower() == "ground":
                if len(tok) != 2:
                    raise NetlistError(f"line {lineno}: 'ground' needs one node")
                ckt.ground = tok[1]

            elif head.lower() in ("open", "open_loop"):
                if len(tok) < 2:
                    raise NetlistError(f"line {lineno}: 'open' needs a loop name")
                ckt.open_loops.extend(tok[1:])

            elif head.lower() in ("flux", "flux_bias"):
                if len(tok) < 2:
                    raise NetlistError(
                        f"line {lineno}: 'flux' needs 'loop [value]' (external flux)")
                ckt.set_flux_bias(tok[1], tok[2] if len(tok) >= 3 else None)

            elif head.lower() in ("offset", "charge", "offset_charge"):
                if len(tok) < 2:
                    raise NetlistError(
                        f"line {lineno}: 'offset' needs 'node [value]' (offset charge)")
                ckt.set_offset_charge(tok[1], tok[2] if len(tok) >= 3 else None)

            elif head.lower() in ("title", "name"):
                ckt.title = " ".join(tok[1:])

            else:
                raise NetlistError(
                    f"line {lineno}: unrecognised entry {head!r} (expected "
                    "C/L/J/QPS, gyrator, loop, ground, open, flux, offset, or title)")
        except NetlistError:
            raise
        except Exception as exc:  # surface circuit-building errors with the line
            raise NetlistError(f"line {lineno}: {exc}") from exc

    return ckt


def from_netlist(source: str) -> Circuit:
    """Build a circuit from a netlist file path or a netlist string."""
    if os.path.exists(source) or source.endswith((".txt", ".net", ".circuit")):
        with open(source, "r") as fh:
            text = fh.read()
    else:
        text = source
    return parse_netlist(text)


def to_netlist(circuit) -> str:
    """Serialise a :class:`~fluxcharge.circuit.Circuit` back to netlist text.

    Inverse of :func:`parse_netlist` (round-trips the element list, gyrators,
    loops and any ``title`` / ``ground`` / ``open`` directives).
    """
    from .elements import (
        Capacitor, Inductor, JosephsonJunction, QuantumPhaseSlip, Gyrator,
    )
    import sympy as sp

    def val(x):
        return sp.sstr(x).replace(" ", "")

    lines = []
    title = getattr(circuit, "title", None)
    if title:
        lines.append(f"title {title}")
    for elem in circuit._elements:
        if isinstance(elem, Capacitor):
            e = elem.edges()[0]
            lines.append(f"C    {e.name}  {e.tail} {e.head}  {val(elem.C)}")
        elif isinstance(elem, Inductor):
            e = elem.edges()[0]
            lines.append(f"L    {e.name}  {e.tail} {e.head}  {val(elem.L)}")
        elif isinstance(elem, JosephsonJunction):
            e = elem.edges()[0]
            lines.append(f"J    {e.name}  {e.tail} {e.head}  {val(elem.EJ)}")
        elif isinstance(elem, QuantumPhaseSlip):
            e = elem.edges()[0]
            lines.append(f"QPS  {e.name}  {e.tail} {e.head}  {val(elem.ES)}")
        elif isinstance(elem, Gyrator):
            e1, e2 = elem.edge1, elem.edge2
            lines.append(f"gyrator  {e1.name} {e1.tail} {e1.head}   "
                         f"{e2.name} {e2.tail} {e2.head}   {val(elem.G)}")
    for loop, entries in circuit._loops.items():
        toks = " ".join((("+" if s > 0 else "-") + en) for s, en in entries)
        lines.append(f"loop  {loop}  {toks}")
    if getattr(circuit, "ground", None):
        lines.append(f"ground {circuit.ground}")
    for ol in getattr(circuit, "open_loops", []) or []:
        lines.append(f"open   {ol}")
    for loop, fx in getattr(circuit, "_flux_bias", {}).items():
        lines.append(f"flux   {loop}  {val(fx)}")
    for node, ng in getattr(circuit, "_offset_charge", {}).items():
        lines.append(f"offset {node}  {val(ng)}")
    return "\n".join(lines) + "\n"

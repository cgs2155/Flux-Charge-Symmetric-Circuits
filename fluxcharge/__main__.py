"""
Command-line workflow: a circuit netlist in, a drawing + Lagrangian +
Hamiltonian out.

    python -m fluxcharge mycircuit.txt
    fluxcharge mycircuit.txt -o mycircuit.png

See :mod:`fluxcharge.netlist` for the netlist format.
"""

from __future__ import annotations

import argparse
import os
import sys

import sympy as sp

from .netlist import from_netlist, NetlistError


def analyze(source, draw=True, outfile=None, canonical=True, show_lagrangian=True,
            make_dual=False):
    """Run the full pipeline on a netlist (path or string) and print results.

    Returns ``(circuit, reduction_result)``.
    """
    ckt = from_netlist(source)
    ckt.validate()
    if make_dual:
        from .transformations import dual
        ckt = dual(ckt)

    title = getattr(ckt, "title", None)
    if title:
        print(f"# {title}")
    print(ckt.summary())

    if show_lagrangian:
        print("\nLagrangian:")
        sp.pprint(ckt.lagrangian())

    ground = getattr(ckt, "ground", None)
    open_loops = getattr(ckt, "open_loops", None) or None
    result = ckt.hamiltonian(ground=ground, open_loops=open_loops,
                             canonical=canonical)
    print("\n" + result.report())

    if draw:
        if outfile is None:
            base = source if not os.path.exists(source) else os.path.splitext(source)[0]
            outfile = (os.path.splitext(source)[0] + ".png"
                       if os.path.exists(source) else "circuit.png")
        # the outer face is a property of the planar embedding, not the gauge;
        # let the schematic auto-detect it so the "open" choice can't distort it
        try:
            ckt.schematic(path=outfile)
            print(f"\nSchematic written to {outfile}")
        except Exception as exc:
            print(f"\n(could not draw schematic: {exc})")

    return ckt, result


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="fluxcharge",
        description="Turn a circuit netlist into a drawing, Lagrangian and Hamiltonian.")
    p.add_argument("netlist", help="path to a circuit netlist (see fluxcharge.netlist)")
    p.add_argument("-o", "--output", help="schematic image path (default: <netlist>.png)")
    p.add_argument("--no-draw", action="store_true", help="skip the schematic")
    p.add_argument("--raw", action="store_true",
                   help="do not canonicalize the Hamiltonian")
    p.add_argument("--no-lagrangian", action="store_true",
                   help="do not print the Lagrangian")
    p.add_argument("--dual", action="store_true",
                   help="analyze the LCG dual circuit instead (C<->L, JJ<->QPS, G->-1/G)")
    args = p.parse_args(argv)

    try:
        analyze(args.netlist, draw=not args.no_draw, outfile=args.output,
                canonical=not args.raw, show_lagrangian=not args.no_lagrangian,
                make_dual=args.dual)
    except (NetlistError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

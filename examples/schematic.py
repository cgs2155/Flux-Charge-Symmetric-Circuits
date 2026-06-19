"""Draw circuits as lumped-element schematics (real symbols, straight wires).

Requires the optional schematic dependency::

    pip install "fluxcharge[schematic]"      # schemdraw + networkx + matplotlib

Run with ``python examples/schematic.py``; PNGs are written next to it.
"""

import matplotlib
matplotlib.use("Agg")

from fluxcharge import Circuit


def lc():
    c = Circuit()
    c.add_inductor("e1", "v1", "v2", L="L")
    c.add_capacitor("e2", "v2", "v1", C="C")
    c.add_loop("f1", ["+e1", "+e2"])
    return c, None


def parallel_caps():
    c = Circuit()
    c.add_inductor("e1", "v1", "v2", L="L")
    c.add_capacitor("e2", "v1", "v2", C="C1")
    c.add_capacitor("e3", "v1", "v2", C="C2")
    for n, e in [("f1", ["+e1", "-e2"]), ("f2", ["+e2", "-e3"]), ("f3", ["-e1", "+e3"])]:
        c.add_loop(n, e)
    return c, None


def circulator():
    c = Circuit()
    c.add_josephson("e1", "v1", "v2", EJ="E_J")
    c.add_capacitor("e2", "v2", "v3", C="C")
    c.add_capacitor("e3", "v3", "v1", C="C")
    c.add_gyrator(("e4", "v1", "v3"), ("e5", "v2", "v3"), G="G")
    for n, e in [("f1", ["+e3", "+e4"]), ("f2", ["+e1", "-e4", "+e5"]),
                 ("f3", ["+e2", "-e5"]), ("f4", ["-e1", "-e2", "-e3"])]:
        c.add_loop(n, e)
    # the planar layout places this automatically from the faces; pass
    # positions={...} only if you want explicit placement.
    return c, None


if __name__ == "__main__":
    for builder, name in [(lc, "lc"), (parallel_caps, "parallel_caps"),
                          (circulator, "circulator")]:
        ckt, positions = builder()
        ckt.schematic(path=f"schematic_{name}.png", positions=positions)
        print(f"wrote schematic_{name}.png")

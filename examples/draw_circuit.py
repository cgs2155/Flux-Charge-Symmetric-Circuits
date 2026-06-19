"""Draw circuits as graphs reconstructed from the boundary matrices A and B.

Requires the optional visualization dependencies::

    pip install "fluxcharge[viz]"        # networkx + matplotlib

Run with ``python examples/draw_circuit.py``; three PNGs are written next to it.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fluxcharge import Circuit


def circulator():
    c = Circuit()
    c.add_josephson("e1", "v1", "v2", EJ="E_J")
    c.add_capacitor("e2", "v2", "v3", C="C")
    c.add_capacitor("e3", "v3", "v1", C="C")
    c.add_gyrator(("e4", "v1", "v3"), ("e5", "v2", "v3"), G="G")
    for n, e in [("f1", ["+e3", "+e4"]), ("f2", ["+e1", "-e4", "+e5"]),
                 ("f3", ["+e2", "-e5"]), ("f4", ["-e1", "-e2", "-e3"])]:
        c.add_loop(n, e)
    return c


def parallel_caps():
    c = Circuit()
    c.add_inductor("e1", "v1", "v2", L="L")
    c.add_capacitor("e2", "v1", "v2", C="C1")
    c.add_capacitor("e3", "v1", "v2", C="C2")
    c.add_loop("f1", ["+e1", "-e2"])
    c.add_loop("f2", ["+e2", "-e3"])
    c.add_loop("f3", ["-e1", "+e3"])
    return c


def lc():
    c = Circuit()
    c.add_inductor("e1", "v1", "v2", L="L")
    c.add_capacitor("e2", "v2", "v1", C="C")
    c.add_loop("f1", ["+e1", "+e2"])
    return c


if __name__ == "__main__":
    for ckt, name, title in [
        (circulator(), "circulator", "Circulator: JJ + 2 caps + gyrator"),
        (parallel_caps(), "parallel_caps", "Parallel capacitors across an inductor"),
        (lc(), "lc", "LC oscillator"),
    ]:
        # the MultiDiGraph is reconstructed purely from A (edges/orientation)
        # and B (faces); it is also the natural data model for a UI builder.
        G = ckt.to_networkx()
        print(f"{name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
              f"{len(G.graph['loops'])} faces")
        ckt.draw(path=f"{name}.png", title=title)
        plt.close("all")
    print("wrote circulator.png, parallel_caps.png, lc.png")

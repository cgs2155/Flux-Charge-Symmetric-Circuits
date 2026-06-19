"""
The non-reciprocal circuit of the manuscript's worked example
=============================================================

A Josephson junction and two capacitors arranged around a single ideal
gyrator (the building block of an active circulator).  This script builds the
circuit, prints the intermediate objects, and reduces it to the Hamiltonian,
reproducing Eq. (eq:hamiltonian) of

    "Gyrators for superconducting circuit design",
    C. Salcedo, S. Cocquyt, A. Osborne, A. A. Houck.

Run with::

    python examples/circulator_jj.py
"""

import sympy as sp

from fluxcharge import Circuit


def build():
    ckt = Circuit()
    # inductive (non-linear) branch: a Josephson junction  v1 -> v2
    ckt.add_josephson("e1", "v1", "v2", EJ="E_J")
    # two identical capacitors
    ckt.add_capacitor("e2", "v2", "v3", C="C")
    ckt.add_capacitor("e3", "v3", "v1", C="C")
    # one ideal gyrator coupling the half-edges (e4) and (e5)
    ckt.add_gyrator(("e4", "v1", "v3"), ("e5", "v2", "v3"), G="G")
    # the four faces of the planar embedding, as signed edge lists
    ckt.add_loop("f1", ["+e3", "+e4"])
    ckt.add_loop("f2", ["+e1", "-e4", "+e5"])
    ckt.add_loop("f3", ["+e2", "-e5"])
    ckt.add_loop("f4", ["-e1", "-e2", "-e3"])
    return ckt


def main():
    ckt = build()
    print(ckt.summary())
    print()

    print("Incidence matrix A (|E| x |V|):")
    sp.pprint(ckt.incidence_matrix())
    print("\nOrientation matrix B (|F| x |E|):")
    sp.pprint(ckt.orientation_matrix())
    print("\nExactness check  B * A = 0 :")
    sp.pprint(ckt.orientation_matrix() * ckt.incidence_matrix())

    print("\nConnection matrix M = (1/2) B (P_C - P_I) A :")
    sp.pprint(ckt.connection_matrix())

    print("\nLagrangian L:")
    sp.pprint(ckt.lagrangian())

    print("\nReducing to the Hamiltonian ...")
    # ground node v1 (global-flux gauge) and open the outer loop f4
    # (global-charge gauge); the cyclic coordinates are found automatically.
    result = ckt.hamiltonian(ground="v1", open_loops="f4")
    print()
    print(result.report())

    # compare with the published result
    G, C, EJ = sp.Symbol("G"), sp.Symbol("C"), sp.Symbol("E_J")
    phi2, q3 = sp.Symbol("phi_v2"), sp.Symbol("q_f3")
    H_paper = (G * phi2 + q3) ** 2 / (2 * C) + q3 ** 2 / (2 * C) - EJ * sp.cos(phi2)
    print("\nPublished Hamiltonian:")
    sp.pprint(H_paper)
    print("\nDifference H - H_published :", sp.expand(result.H - H_paper))


if __name__ == "__main__":
    main()

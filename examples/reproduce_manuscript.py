#!/usr/bin/env python3
"""
Reproduce the worked example of

    "Gyrators for superconducting circuit design",
    C. Salcedo, S. Cocquyt, A. Osborne, A. A. Houck.

Builds the manuscript's circulator circuit (a Josephson junction, two
capacitors and a gyrator), prints the boundary matrices, the flux-charge
symmetric Lagrangian and the reduced Hamiltonian, and checks the Hamiltonian
against the published closed form (Eq. eq:hamiltonian).  With the optional
schematic dependency it also regenerates the circuit diagram.

Run:  python examples/reproduce_manuscript.py
"""

import sympy as sp

from fluxcharge import Circuit


def build():
    ckt = Circuit()
    ckt.add_josephson("e1", "v1", "v2", EJ="E_J")
    ckt.add_capacitor("e2", "v2", "v3", C="C")
    ckt.add_capacitor("e3", "v3", "v1", C="C")
    ckt.add_gyrator(("e4", "v1", "v3"), ("e5", "v2", "v3"), G="G")
    ckt.add_loop("f1", ["+e3", "+e4"])
    ckt.add_loop("f2", ["+e1", "-e4", "+e5"])
    ckt.add_loop("f3", ["+e2", "-e5"])
    ckt.add_loop("f4", ["-e1", "-e2", "-e3"])
    return ckt


def main():
    ckt = build()
    ckt.validate()

    print("Incidence matrix A (|E| x |V|):")
    sp.pprint(ckt.incidence_matrix())
    print("\nLoop matrix B (|F| x |E|):")
    sp.pprint(ckt.orientation_matrix())
    print("\nAntisymmetric form Omega:")
    sp.pprint(ckt.omega())

    print("\nFlux-charge symmetric Lagrangian:")
    sp.pprint(ckt.lagrangian())

    result = ckt.hamiltonian(ground="v1", open_loops="f4")
    print("\n" + result.report())

    G, C, EJ = sp.symbols("G C E_J")
    phi2, q3 = sp.symbols("phi_v2 q_f3")
    H_published = (G * phi2 + q3) ** 2 / (2 * C) + q3 ** 2 / (2 * C) - EJ * sp.cos(phi2)

    diff = sp.expand(result.H - H_published)
    print("\nPublished Hamiltonian (Eq. eq:hamiltonian):")
    sp.pprint(H_published)
    print(f"\nH(package) - H(published) = {diff}")
    assert diff == 0, "MISMATCH with the published Hamiltonian"
    assert result.complete
    print("\n*** Reproduced the manuscript Hamiltonian exactly. ***")

    try:
        ckt.schematic(path="manuscript_circulator.png")
        print("Wrote schematic to manuscript_circulator.png")
    except Exception as exc:  # schematic deps optional
        print(f"(schematic skipped: {exc})")


if __name__ == "__main__":
    main()

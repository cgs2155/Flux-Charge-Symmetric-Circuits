"""
A bare LC oscillator
====================

The simplest sanity check: one inductor and one capacitor in a single loop.
With node ``v1`` grounded the Hamiltonian must be

    H = q^2 / (2 C)  +  phi^2 / (2 L),

the textbook harmonic oscillator in conjugate flux/charge variables.

Run with::

    python examples/lc_oscillator.py
"""

import sympy as sp

from fluxcharge import Circuit


def main():
    ckt = Circuit()
    ckt.add_inductor("e1", "v1", "v2", L="L")
    ckt.add_capacitor("e2", "v2", "v1", C="C")
    ckt.add_loop("f1", ["+e1", "+e2"])

    print(ckt.summary())
    print("\nLagrangian:")
    sp.pprint(ckt.lagrangian())

    result = ckt.hamiltonian(ground="v1")
    print("\n" + result.report())


if __name__ == "__main__":
    main()

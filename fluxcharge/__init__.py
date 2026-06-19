"""
fluxcharge
==========

Symbolic construction of Hamiltonians for lumped-element LCG circuits
(inductors, capacitors, Josephson junctions, quantum phase slips, gyrators)
following the flux-charge symmetric formalism of

    "Gyrators for superconducting circuit design",
    C. Salcedo, S. Cocquyt, A. Osborne, A. A. Houck.

Typical use::

    from fluxcharge import Circuit

    ckt = Circuit()
    ckt.add_josephson("e1", "v1", "v2", EJ="E_J")
    ckt.add_capacitor("e2", "v2", "v3", C="C")
    ckt.add_capacitor("e3", "v3", "v1", C="C")
    ckt.add_gyrator(("e4", "v1", "v3"), ("e5", "v2", "v3"), G="G")
    ckt.add_loop("f1", ["+e3", "+e4"])
    ckt.add_loop("f2", ["+e1", "-e4", "+e5"])
    ckt.add_loop("f3", ["+e2", "-e5"])
    ckt.add_loop("f4", ["-e1", "-e2", "-e3"])

    L = ckt.lagrangian()
    result = ckt.hamiltonian()      # symbolic reduction
    print(result.H)
"""

from .circuit import Circuit
from .elements import (
    Capacitor,
    Gyrator,
    Inductor,
    JosephsonJunction,
    QuantumPhaseSlip,
)
from .reduction import Reducer, ReductionResult
from .visualize import circuit_to_networkx, draw_circuit
from .schematic import draw_schematic
from .netlist import from_netlist, parse_netlist, to_netlist
from .transformations import dual

__all__ = [
    "Circuit",
    "Capacitor",
    "Inductor",
    "JosephsonJunction",
    "QuantumPhaseSlip",
    "Gyrator",
    "Reducer",
    "ReductionResult",
    "circuit_to_networkx",
    "draw_circuit",
    "draw_schematic",
    "from_netlist",
    "parse_netlist",
    "to_netlist",
    "dual",
]

__version__ = "0.1.0"

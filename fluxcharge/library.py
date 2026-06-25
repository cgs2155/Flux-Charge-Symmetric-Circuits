"""
A small library of ready-made circuits.

Each constructor returns a fully wired :class:`~fluxcharge.circuit.Circuit`
(elements, loops, gauge and any external bias already set), so you can go
straight to ``.hamiltonian()`` and, with :meth:`Circuit.natural_params`, to a
spectrum in GHz.  Standard symbol names (``C``, ``L``, ``E_J``, ``E_S``, ``G``,
and bias symbols ``n_g_*`` / ``phi_ext_*``) are used so physical values drop in
cleanly.

Besides the textbook qubits this includes the showcases unique to the
flux-charge formalism: the manuscript's gyrator **circulator**, and a quantum
**phase-slip** qubit (the charge-space dual of a transmon/fluxonium).
"""

from __future__ import annotations

from .circuit import Circuit


def lc_resonator() -> Circuit:
    """Bare LC oscillator (symbols ``L``, ``C``).  Mode: EXTENDED."""
    c = Circuit()
    c.title = "LC resonator"
    c.add_inductor("e1", "v1", "v2", L="L")
    c.add_capacitor("e2", "v2", "v1", C="C")
    c.add_loop("f1", ["+e1", "+e2"])
    c.ground = "v1"
    return c


def transmon() -> Circuit:
    """Transmon: a Josephson junction shunted by a capacitor (``E_J``, ``C``).
    Mode: PERIODIC."""
    c = Circuit()
    c.title = "Transmon"
    c.add_josephson("e1", "v1", "v2", EJ="E_J")
    c.add_capacitor("e2", "v1", "v2", C="C")
    c.add_loop("f1", ["+e1", "-e2"])
    c.ground = "v1"
    return c


def cooper_pair_box() -> Circuit:
    """Cooper-pair box: a transmon with a gate/offset charge ``n_g_v2`` on the
    island (sweep it for the charge dispersion)."""
    c = transmon()
    c.title = "Cooper-pair box"
    c.set_offset_charge("v2")
    return c


def fluxonium(flux_bias: bool = True) -> Circuit:
    """Fluxonium: a junction shunted by a (super)inductor and a capacitor
    (``E_J``, ``L``, ``C``).  With *flux_bias* an external flux ``phi_ext_f1``
    threads the JJ-inductor loop (sweet spot at ``pi``)."""
    c = Circuit()
    c.title = "Fluxonium"
    c.add_josephson("e1", "v1", "v2", EJ="E_J")
    c.add_inductor("e2", "v1", "v2", L="L")
    c.add_capacitor("e3", "v1", "v2", C="C")
    c.add_loop("f1", ["+e1", "-e2"])
    c.add_loop("f2", ["+e2", "-e3"])
    c.add_loop("f3", ["-e1", "+e3"])
    c.ground = "v1"
    c.open_loops = ["f3"]
    if flux_bias:
        c.set_flux_bias("f1")
    return c


def phase_slip_qubit(charge_bias: bool = True) -> Circuit:
    """Quantum phase-slip qubit: a coherent QPS shunting an inductor (``E_S``,
    ``L``) -- the LCG dual of a transmon/fluxonium.  Mode: DUAL_PERIODIC (the
    charge is compact, the flux is the integer fluxoid).  With *charge_bias* an
    offset charge ``n_g_v2`` is applied (the dual of an external flux)."""
    c = Circuit()
    c.title = "Phase-slip qubit"
    c.add_qps("e1", "v1", "v2", ES="E_S")
    c.add_inductor("e2", "v1", "v2", L="L")
    c.add_loop("f1", ["+e1", "-e2"])
    c.ground = "v1"
    if charge_bias:
        c.set_offset_charge("v2")
    return c


def zero_pi() -> Circuit:
    """The 0-pi qubit: two Josephson junctions (with junction capacitances), two
    superinductors and two large cross-capacitors (``E_J``, ``C_J``, ``L``,
    ``C``).

    A protected qubit and the canonical multi-mode showcase.  The node frame
    used here -- both junctions meeting at ``v3``, both inductors at ``v1``,
    cross-capacitors on ``v1-v3`` and ``v2-v4`` -- is the one in which the
    circuit's compact mode is **manifest**: it reduces to three modes, one
    *periodic* (the junction phase ``phi_v3`` lives only inside cosines, with
    integer coefficients) and two *extended*.  Because the compact coordinate is
    aligned with the frame, the spectrum diagonalizes cleanly (no hidden-compact
    / ``cos(theta/2)`` lattice obstruction).  Ships with a default schematic
    layout (an equilateral triangle ``v1, v2, v3`` with ``v4`` at the centre).
    """
    c = Circuit()
    c.title = "Zero-pi"
    c.add_josephson("j1", "v2", "v3", EJ="E_J")
    c.add_capacitor("cJ1", "v2", "v3", C="C_J")
    c.add_josephson("j2", "v4", "v3", EJ="E_J")
    c.add_capacitor("cJ2", "v4", "v3", C="C_J")
    c.add_inductor("l1", "v1", "v2", L="L")
    c.add_inductor("l2", "v1", "v4", L="L")
    c.add_capacitor("c1", "v1", "v3", C="C")        # cross-capacitor
    c.add_capacitor("c2", "v2", "v4", C="C")        # cross-capacitor
    c.ground = "v1"
    # preferred schematic layout: triangle (v1, v2, v3) with v4 at the centre
    c._positions = {"v1": (-5.16, -3.0), "v2": (0.0, 6.0),
                    "v3": (5.16, -3.0), "v4": (0.0, 0.0)}
    return c


def circulator() -> Circuit:
    """The manuscript's non-reciprocal circulator: a Josephson junction and two
    capacitors coupled by a gyrator (``E_J``, ``C``, ``G``)."""
    c = Circuit()
    c.title = "Circulator"
    c.add_josephson("e1", "v1", "v2", EJ="E_J")
    c.add_capacitor("e2", "v2", "v3", C="C")
    c.add_capacitor("e3", "v3", "v1", C="C")
    c.add_gyrator(("e4", "v1", "v3"), ("e5", "v2", "v3"), G="G")
    for n, e in [("f1", ["+e3", "+e4"]), ("f2", ["+e1", "-e4", "+e5"]),
                 ("f3", ["+e2", "-e5"]), ("f4", ["-e1", "-e2", "-e3"])]:
        c.add_loop(n, e)
    c.ground = "v1"
    c.open_loops = ["f4"]
    return c


#: name -> constructor, for discovery / iteration
CIRCUITS = {
    "lc_resonator": lc_resonator,
    "transmon": transmon,
    "cooper_pair_box": cooper_pair_box,
    "fluxonium": fluxonium,
    "phase_slip_qubit": phase_slip_qubit,
    "zero_pi": zero_pi,
    "circulator": circulator,
}

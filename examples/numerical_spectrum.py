"""
Numerical diagonalization of reduced flux-charge Hamiltonians.

Builds the transmon, fluxonium and the manuscript circulator, classifies their
modes (mode-type detection), diagonalizes them numerically, and -- if matplotlib
is available -- saves a few plots.  Requires the optional ``numpy`` dependency
(``pip install "fluxcharge[numeric]"``); plotting uses the always-installed
matplotlib.
"""

import numpy as np

from fluxcharge import Circuit


def transmon():
    c = Circuit()
    c.add_josephson("e1", "v1", "v2", EJ="E_J")
    c.add_capacitor("e2", "v1", "v2", C="C")
    c.add_loop("f1", ["+e1", "-e2"])
    return c.hamiltonian(ground="v1")


def fluxonium():
    c = Circuit()
    c.add_josephson("e1", "v1", "v2", EJ="E_J")
    c.add_inductor("e2", "v1", "v2", L="L")
    c.add_capacitor("e3", "v1", "v2", C="C")
    c.add_loop("f1", ["+e1", "-e2"])
    c.add_loop("f2", ["+e2", "-e3"])
    c.add_loop("f3", ["-e1", "+e3"])
    return c.hamiltonian(ground="v1", open_loops="f3")


def circulator():
    c = Circuit()
    c.add_josephson("e1", "v1", "v2", EJ="E_J")
    c.add_capacitor("e2", "v2", "v3", C="C")
    c.add_capacitor("e3", "v3", "v1", C="C")
    c.add_gyrator(("e4", "v1", "v3"), ("e5", "v2", "v3"), G="G")
    for n, e in [("f1", ["+e3", "+e4"]), ("f2", ["+e1", "-e4", "+e5"]),
                 ("f3", ["+e2", "-e5"]), ("f4", ["-e1", "-e2", "-e3"])]:
        c.add_loop(n, e)
    return c.hamiltonian(ground="v1", open_loops="f4")


if __name__ == "__main__":
    tr = transmon()
    print("transmon  H =", tr.H)
    print("  modes:", tr.modes())
    ev = tr.eigenenergies({"E_J": 15.0, "C": 1.0}, n_levels=5, cutoffs={"q_f1": 81})
    print("  levels:", np.round(ev, 4))
    print("  E_01   :", round(float(ev[1] - ev[0]), 4))

    fx = fluxonium()
    print("\nfluxonium H =", fx.H)
    print("  modes:", fx.modes())
    print("  levels:", np.round(
        fx.eigenenergies({"E_J": 10.0, "L": 1.0, "C": 1.0}, n_levels=5,
                         cutoffs={"phi_v2": 60}), 4))

    cir = circulator()
    print("\ncirculator H =", cir.canonical().H)
    print("  modes:", cir.modes())
    print("  levels:", np.round(
        cir.eigenenergies({"E_J": 10.0, "C": 1.0, "G": 0.7}, n_levels=5,
                          cutoffs={"phi_v2": 80}), 4))

    try:
        import matplotlib
        matplotlib.use("Agg")
        tr.plot_potential_wavefunctions({"E_J": 15.0, "C": 1.0},
                                        cutoffs={"q_f1": 61}, path="transmon_wf.png")
        fx.plot_potential_wavefunctions({"E_J": 10.0, "L": 1.0, "C": 1.0},
                                        cutoffs={"phi_v2": 60}, path="fluxonium_wf.png")
        tr.plot_spectrum("q_f1", np.linspace(-1, 1, 41), {"E_J": 1.0, "C": 1.0},
                         n_levels=4, cutoffs={"q_f1": 61}, relative=True,
                         path="transmon_dispersion.png")
        print("\nwrote transmon_wf.png, fluxonium_wf.png, transmon_dispersion.png")
    except Exception as exc:
        print(f"\n(plotting skipped: {exc})")

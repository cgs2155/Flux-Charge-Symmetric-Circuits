"""
Tour of the built-in circuit library, in physical units (GHz).

Builds each ready-made circuit, prints its Hamiltonian, and reports a spectrum
in GHz -- including the gyrator circulator and the quantum phase-slip qubit that
are unique to the flux-charge formalism.  Requires the optional ``numpy``
(``pip install "fluxcharge[numeric]"``).
"""

import numpy as np

from fluxcharge import library


def show(title, line):
    print(f"\n=== {title} ===")
    print(line)


if __name__ == "__main__":
    # --- transmon: frequency + anharmonicity in GHz, charge matrix element ---
    tr = library.transmon()
    res = tr.hamiltonian(ground="v1")
    p = tr.natural_params({"C": "70fF", "E_J": "15GHz"})
    ev = res.eigenenergies(p, n_levels=4, cutoffs={"q_f1": 81})
    show("Transmon (C=70 fF, E_J=15 GHz)", res.H)
    print(f"  f01 = {ev[1]-ev[0]:.4f} GHz   anharmonicity = "
          f"{(ev[2]-ev[1])-(ev[1]-ev[0]):.4f} GHz")
    M = res.matrix_elements("q_f1", p, n_levels=3, cutoffs={"q_f1": 81})
    print(f"  |<0|n|1>| = {abs(M[0,1]):.4f}")

    # --- fluxonium: flux sweep + sweet-spot insensitivity ---
    fx = library.fluxonium()
    rf = fx.hamiltonian(ground="v1", open_loops="f3")
    # capacitance/inductance given directly as energies (E_C, E_L) in GHz
    pf = fx.natural_params({"E_J": "5GHz", "C": "1GHz", "L": "1GHz"})
    show("Fluxonium (E_J=5, E_C=1, E_L=1 GHz)", rf.H)
    import math
    for px in (0.0, math.pi):
        pp = dict(pf); pp["phi_ext_f1"] = px
        e = rf.eigenenergies(pp, n_levels=2, cutoffs={"phi_v2": 60})
        df, _ = rf.transition_sensitivity("phi_ext_f1", pp, cutoffs={"phi_v2": 60})
        print(f"  phi_ext={px:.3f}: f01={e[1]-e[0]:.4f} GHz   df01/dphi={df:+.4f}")

    # --- phase-slip qubit: the LCG dual of the transmon ---
    qp = library.phase_slip_qubit(charge_bias=False)
    rq = qp.hamiltonian(ground="v1")
    ev_q = rq.eigenenergies({"E_S": 10.0, "L": 1.0}, n_levels=4, cutoffs={"phi_v2": 80})
    ev_t = library.transmon().hamiltonian(ground="v1").eigenenergies(
        {"E_J": 10.0, "C": 1.0}, n_levels=4, cutoffs={"q_f1": 81})
    show("Phase-slip qubit vs transmon (duality)", rq.H)
    print(f"  max |level difference| = {np.max(np.abs(ev_q-ev_t)):.2e}  (should be ~0)")

    # --- zero-pi: a 3-mode protected qubit ---
    zp = library.zero_pi()
    rz = zp.hamiltonian(ground="n1", strict=False, canonical=True)
    show("Zero-pi (3 modes)", rz.H)
    print("  modes (auto-classified):", [(str(m.flux), str(m.charge), m.kind)
                                          for m in rz.modes()])
    cut = {str(b): 6 for _a, b, _c in rz.conjugate_pairs}     # small basis (not converged)
    evz = rz.eigenenergies({"E_J": 10.0, "C_J": 1.0, "L": 1.0, "C": 1.0},
                           n_levels=4, cutoffs=cut)
    print("  spectrum (cutoff 6/mode -- NOT converged; raise cutoffs to refine):",
          np.round(evz - evz[0], 4))

    # --- circulator: non-reciprocal, gyrator cross term ---
    cir = library.circulator()
    rc = cir.hamiltonian(ground="v1", open_loops="f4", canonical=True)
    ev_c = rc.eigenenergies({"E_J": 10.0, "C": 1.0, "G": 0.5}, n_levels=4,
                            cutoffs={"phi_v2": 60})
    show("Circulator (E_J=10, C=1, G=0.5)", rc.H)
    print(f"  levels: {np.round(ev_c, 4)}")

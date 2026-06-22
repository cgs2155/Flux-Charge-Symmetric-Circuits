"""
Import scqubits circuits and cross-validate, then show the extra reach.

Imports scqubits' branch-YAML into fluxcharge and checks the spectrum matches
scqubits' trusted predefined qubit classes (Transmon, Fluxonium) -- then adds a
quantum phase slip, which scqubits cannot represent.

Requires both ``scqubits`` and ``numpy`` (``pip install scqubits``).  Note: a
junction's textbook charging energy is ``E_C = e**2/2C``; scqubits' predefined
classes use that convention and the importer follows it (matching them exactly).
scqubits' own *custom Circuit* YAML uses a different internal normalisation, so
compare against the predefined classes, which are the textbook reference.
"""

import numpy as np

try:
    import scqubits
except ImportError:
    raise SystemExit("this demo needs scqubits:  pip install scqubits")

from fluxcharge import from_scqubits_yaml


def rel(ev):
    ev = np.asarray(ev, dtype=float)
    return ev - ev[0]


def fc_spectrum(yaml, n=5, cut=160):
    ckt, params = from_scqubits_yaml(yaml)
    res = ckt.hamiltonian(ground=ckt.ground or "1", strict=False, canonical=True)
    cutoffs = {str(b): cut for _a, b, _c in res.conjugate_pairs}
    return rel(res.eigenenergies(params, n_levels=n, cutoffs=cutoffs)), res


if __name__ == "__main__":
    # --- transmon: import vs scqubits.Transmon ---
    ev_fc, _ = fc_spectrum("branches:\n- [JJ, 1, 0, 15]\n- [C, 1, 0, 0.3]\n", cut=101)
    ev_sc = rel(scqubits.Transmon(EJ=15.0, EC=0.3, ng=0.0, ncut=60).eigenvals(5))
    print("Transmon   import:", np.round(ev_fc, 4))
    print("        scqubits :", np.round(ev_sc, 4))
    print("        max|diff|:", np.max(np.abs(ev_fc - ev_sc)))

    # --- fluxonium: import vs scqubits.Fluxonium ---
    ev_fc, _ = fc_spectrum("branches:\n- [JJ,1,2,10,1e15]\n- [L,1,2,0.1]\n- [C,1,2,0.2]\n")
    ev_sc = rel(scqubits.Fluxonium(EJ=10.0, EC=0.2, EL=0.1, flux=0.0, cutoff=110).eigenvals(5))
    print("\nFluxonium  import:", np.round(ev_fc, 4))
    print("        scqubits :", np.round(ev_sc, 4))
    print("        max|diff|:", np.max(np.abs(ev_fc - ev_sc)))

    # --- the extra reach: take the imported circuit and add a QPS (scqubits can't) ---
    ckt, params = from_scqubits_yaml("branches:\n- [L, 1, 2, EL]\n")
    ckt.add_qps("qps", "1", "2", ES="E_S")     # a coherent phase slip across the inductor
    res = ckt.hamiltonian(ground="1", strict=False, canonical=True)
    print("\nImported an inductor, then added a QPS -> phase-slip qubit:")
    print("  H =", res.H, " (scqubits has no phase-slip element)")

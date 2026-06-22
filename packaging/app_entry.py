"""PyInstaller entry point: import the package (so relative imports work) and
launch the GUI.  Build with:  pyinstaller fluxcharge-gui.spec

Set FLUXCHARGE_SELFTEST=1 to exercise the lazily-imported numeric/coherence code
paths (which the GUI only touches on Diagonalize) and exit -- a smoke test that a
frozen build's compressed modules all decompress, run, then launch the GUI.
"""
import os
import sys


def _selftest():
    import matplotlib
    matplotlib.use("Agg")
    from fluxcharge import library
    res = library.transmon().hamiltonian(ground="v1")
    ev = res.eigenenergies({"E_J": 15.0, "C": 1.0}, n_levels=3, cutoffs={"q_f1": 41})
    res.matrix_elements("q_f1", {"E_J": 15.0, "C": 1.0}, n_levels=3)
    res.plot_potential_wavefunctions({"E_J": 15.0, "C": 1.0}, n_levels=3,
                                     cutoffs={"q_f1": 41})
    print("SELFTEST OK; levels:", ev[:3])


if __name__ == "__main__":
    if os.environ.get("FLUXCHARGE_SELFTEST"):
        _selftest()
        sys.exit(0)
    from fluxcharge.gui import main
    main()

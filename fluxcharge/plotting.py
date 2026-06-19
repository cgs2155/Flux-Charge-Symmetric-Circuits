"""
Plotting helpers for the numerical spectra of reduced flux-charge circuits.

All functions take a :class:`~fluxcharge.reduction.ReductionResult` (the symbolic
Hamiltonian) plus numeric ``params`` and return a matplotlib ``Axes``; pass
``path=`` to also save the figure.  They build on :mod:`fluxcharge.numerics`.

* :func:`plot_energy_levels` -- the lowest eigenenergies as a level diagram.
* :func:`plot_spectrum` -- eigenenergies as one parameter (a circuit symbol or an
  offset charge) is swept.
* :func:`plot_potential_wavefunctions` -- for a single-mode circuit, the
  potential with the eigenstate probability densities stacked at their energies.
"""

from __future__ import annotations

from typing import Optional, Sequence

import sympy as sp

from . import numerics as _num


def _ax(ax, figsize=(6, 4)):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _save(ax, path, dpi=160):
    if path is not None:
        ax.figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return ax


def plot_energy_levels(result, params=None, n_levels=8, cutoffs=None,
                       offsets=None, mode_types=None, ax=None, path=None,
                       relative=False):
    """Draw the lowest ``n_levels`` eigenenergies as horizontal levels."""
    ax = _ax(ax, figsize=(3.2, 4.5))
    ev = _num.eigenenergies(result, params, n_levels, cutoffs, offsets, mode_types)
    if relative:
        ev = ev - ev[0]
    for i, e in enumerate(ev):
        ax.hlines(e, 0.1, 0.9, color="C0")
        ax.text(0.95, e, f"  {i}", va="center", fontsize=8)
    ax.set_xticks([])
    ax.set_ylabel("energy" + (" (rel. to ground)" if relative else ""))
    ax.set_title("Spectrum")
    return _save(ax, path)


def plot_spectrum(result, parameter, values, params=None, n_levels=6,
                  cutoffs=None, offsets=None, mode_types=None, relative=False,
                  ax=None, path=None):
    """Plot eigenenergies versus a swept *parameter* over *values*.

    *parameter* may be a circuit symbol (e.g. ``"E_J"``) or an offset-charge
    symbol (e.g. ``"q_f1"`` for the transmon charge dispersion).
    """
    import numpy as np
    ax = _ax(ax)
    values = np.asarray(values, dtype=float)
    spec = _num.sweep(result, parameter, values, params, n_levels, cutoffs,
                      offsets, mode_types, relative=relative)
    for k in range(spec.shape[1]):
        ax.plot(values, spec[:, k], label=f"{k}")
    ax.set_xlabel(f"${sp.latex(sp.sympify(parameter))}$")
    ax.set_ylabel("energy" + (" (rel. to ground)" if relative else ""))
    ax.set_title("Spectrum vs " + str(parameter))
    ax.legend(title="level", fontsize=8, ncol=2)
    return _save(ax, path)


def plot_potential_wavefunctions(result, params=None, n_levels=5, cutoffs=None,
                                 offsets=None, mode_types=None, ax=None,
                                 path=None, grid=400, scale=None):
    """Potential and stacked eigenstate densities for a *single-mode* circuit.

    The probability density of each eigenstate is drawn at the height of its
    eigenenergy on top of the potential, in the natural coordinate of the mode
    (flux for ``EXTENDED``/``PERIODIC``, charge for ``DUAL_PERIODIC``).  Raises
    if the reduced circuit has more than one degree of freedom.
    """
    import numpy as np
    result = result if result.is_canonical else result.canonical()
    modes = _num.classify_modes(result, mode_types)
    if len(modes) != 1:
        raise ValueError(
            f"potential/wavefunction plot is for a single-mode circuit; this one "
            f"has {len(modes)} modes. Use plot_energy_levels / plot_spectrum.")
    mode = modes[0]
    ax = _ax(ax)
    x, V, energies, psis = _realspace_1d(result, mode, params, n_levels, cutoffs,
                                         offsets, mode_types, grid)
    ax.plot(x, V, color="k", lw=1.5, zorder=5)
    span = (V.max() - V.min()) or 1.0
    if scale is None:
        gaps = np.diff(energies)
        scale = 0.7 * (np.median(gaps) if len(gaps) and np.median(gaps) > 0
                       else 0.1 * span)
    for i in range(len(energies)):
        dens = np.abs(psis[:, i]) ** 2
        dens = dens / (dens.max() or 1.0) * scale
        ax.axhline(energies[i], color=f"C{i}", lw=0.6, ls=":", alpha=0.6)
        ax.fill_between(x, energies[i], energies[i] + dens, color=f"C{i}",
                        alpha=0.5, label=f"{i}")
    is_charge = mode.kind == _num.DUAL_PERIODIC
    ax.set_xlabel(f"${sp.latex(mode.charge if is_charge else mode.flux)}$")
    ax.set_ylabel("energy")
    ax.set_title("Potential and eigenstates")
    ax.legend(title="level", fontsize=8, ncol=2)
    return _save(ax, path)


def _realspace_1d(result, mode, params, n_levels, cutoffs, offsets, mode_types,
                  grid):
    """Real-space potential, energies and wavefunctions for one mode.

    ``EXTENDED`` modes use a Fourier (Colbert-Miller) DVR flux grid -- which has
    no fermion-doubling artifact -- so the wavefunctions are smooth.
    ``PERIODIC``/``DUAL_PERIODIC`` modes are solved in the integer basis and the
    eigenvectors Fourier-transformed to the conjugate angle.
    """
    import numpy as np
    H = sp.expand(result.H)
    params = {sp.sympify(k): complex(v) for k, v in (params or {}).items()}
    missing = H.free_symbols - {mode.flux, mode.charge} - set(params)
    if missing:
        raise ValueError("missing parameter values: "
                         + ", ".join(sorted(map(str, missing))))
    Hn = sp.expand(H.subs(params))

    if mode.kind == _num.EXTENDED:
        pos, mom = mode.flux, mode.charge
        c2 = complex(Hn.coeff(mom, 2)).real          # kinetic q^2 coefficient
        Vexpr = sp.expand(Hn - Hn.coeff(mom, 2) * mom ** 2)
        if Vexpr.coeff(mom, 1) != 0 or mom in Vexpr.free_symbols:
            raise NotImplementedError(
                f"the Hamiltonian has a momentum-dependent term in {mom} beyond "
                f"the kinetic {mom}**2 (e.g. a gyrator's phi*q cross term), so it "
                "does not split into kinetic + scalar potential V(phi); there is no "
                "1-D potential curve to draw. Use plot_energy_levels / plot_spectrum "
                "instead -- the eigenenergies themselves are unaffected.")
        # half-width from the harmonic length, generously padded
        aphi = float(complex(Hn.coeff(pos, 2)).real) or 1.0
        L = 8.0 * (c2 / aphi) ** 0.25 if aphi > 0 else 12.0
        x = np.linspace(-L, L, grid)
        dx = x[1] - x[0]
        K = _cm_kinetic(grid, dx)                    # approximates -d^2/dx^2
        Vfun = sp.lambdify(pos, Vexpr, "numpy")
        V = np.real(Vfun(x)) * np.ones_like(x)
        Hmat = c2 * K + np.diag(V)
    else:  # PERIODIC / DUAL_PERIODIC: angle = the compact coordinate
        angle, integer = mode.compact, mode.discrete
        # potential as a function of the angle (drop the integer kinetic part)
        Vexpr = sp.expand(Hn - Hn.coeff(integer, 2) * integer ** 2)
        Vfun = sp.lambdify(angle, Vexpr, "numpy")
        x = np.linspace(-np.pi, np.pi, grid)
        V = np.real(Vfun(x)) * np.ones_like(x)
        _, vecs = _num.eigensystem(result, {str(k): v for k, v in params.items()},
                                   n_levels, cutoffs, offsets, mode_types)
        ncut = (vecs.shape[0] - 1) // 2
        ns = np.arange(-ncut, ncut + 1)
        # psi(angle) = sum_n c_n e^{i n angle}
        basis = np.exp(1j * np.outer(x, ns)) / np.sqrt(2 * np.pi)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            psis = basis @ vecs
        energies = _num.eigenenergies(result,
                                      {str(k): v for k, v in params.items()},
                                      n_levels, cutoffs, offsets, mode_types)
        return x, V, energies, psis

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        w, v = np.linalg.eigh(0.5 * (Hmat + Hmat.conj().T))
    n = min(n_levels, len(w))
    return x, V, w[:n], v[:, :n]


def _cm_kinetic(n, dx):
    """Colbert-Miller DVR matrix approximating ``-d^2/dx^2`` on a uniform grid."""
    import numpy as np
    i = np.arange(n)
    diff = i[:, None] - i[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        K = 2.0 * (-1.0) ** diff / np.where(diff == 0, 1, diff) ** 2
    np.fill_diagonal(K, np.pi ** 2 / 3.0)
    return K / dx ** 2

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
                  ax=None, path=None, quantity="levels"):
    """Plot a spectral quantity versus a swept *parameter* over *values*.

    *parameter* may be a circuit symbol (e.g. ``"E_J"``) or an offset-charge
    symbol (e.g. ``"q_f1"`` for the transmon charge dispersion).  *quantity*
    selects what is plotted (all derived from the same diagonalizations):

    * ``"levels"`` -- the eigenenergies ``E_j`` (with *relative* subtracting the
      ground state);
    * ``"transitions"`` -- the transition frequencies ``f_{0->j} = E_j - E_0``
      from the ground state (the trivial ``j=0`` line is dropped);
    * ``"anharmonicity"`` -- ``alpha = (E_2 - E_1) - (E_1 - E_0)`` (needs >= 3
      levels), the quantity that sets gate leakage.
    """
    import numpy as np
    ax = _ax(ax)
    values = np.asarray(values, dtype=float)
    spec = _num.sweep(result, parameter, values, params, n_levels, cutoffs,
                      offsets, mode_types, relative=False)
    xlabel = f"${sp.latex(sp.sympify(parameter))}$"

    if quantity == "transitions":
        for k in range(1, spec.shape[1]):
            ax.plot(values, spec[:, k] - spec[:, 0], label=f"0→{k}")
        ax.set_ylabel("transition frequency")
        ax.set_title(f"Transitions vs {parameter}")
        ax.legend(title="transition", fontsize=8, ncol=2)
    elif quantity == "anharmonicity":
        if spec.shape[1] < 3:
            raise ValueError("anharmonicity needs at least 3 levels")
        alpha = (spec[:, 2] - spec[:, 1]) - (spec[:, 1] - spec[:, 0])
        ax.plot(values, alpha, color="C3")
        ax.axhline(0.0, color="0.7", lw=0.6, ls=":")
        ax.set_ylabel(r"anharmonicity $\alpha = f_{12}-f_{01}$")
        ax.set_title(f"Anharmonicity vs {parameter}")
    else:                                              # "levels"
        base = spec[:, 0] if relative else 0.0
        for k in range(spec.shape[1]):
            ax.plot(values, spec[:, k] - base, label=f"{k}")
        ax.set_ylabel("energy" + (" (rel. to ground)" if relative else ""))
        ax.set_title(f"Spectrum vs {parameter}")
        ax.legend(title="level", fontsize=8, ncol=2)
    ax.set_xlabel(xlabel)
    return _save(ax, path)


def plot_potential_wavefunctions(result, params=None, n_levels=5, cutoffs=None,
                                 offsets=None, mode_types=None, ax=None,
                                 path=None, grid=400, scale=None,
                                 representation="auto"):
    """Eigenstate densities for a *single-mode* circuit, in flux or charge space.

    *representation* selects which variable the wavefunctions are shown in:

    * ``"auto"`` (default) -- the natural *potential* coordinate (flux for a
      junction, charge for a phase slip), drawn over the potential ``V``;
    * ``"flux"`` / ``"charge"`` -- the chosen variable.  When it is the potential
      coordinate you also get ``V``; when it is the conjugate, the densities are
      the Fourier transform of the eigenstates (continuous for an extended mode,
      the discrete charge/flux-number distribution for a periodic one) and there
      is no potential curve (it does not live in that variable).

    Raises if the reduced circuit has more than one degree of freedom.
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

    x, V, energies, psis, pos = _realspace_1d(result, mode, params, n_levels,
                                              cutoffs, offsets, mode_types, grid)
    conj = mode.charge if pos == mode.flux else mode.flux
    target = {"auto": pos, "flux": mode.flux, "charge": mode.charge}[representation]

    discrete = False
    if target == pos:                       # potential representation
        xs, dens_list, xsym, Vcurve = x, [np.abs(psis[:, i]) ** 2
                                          for i in range(len(energies))], pos, V
    elif mode.kind == _num.EXTENDED:        # conjugate of an extended mode: FT
        dx = x[1] - x[0]
        xs = np.fft.fftshift(np.fft.fftfreq(len(x), d=dx)) * 2 * np.pi
        ft = np.fft.fftshift(np.fft.fft(psis, axis=0), axes=0)
        dens_list = [np.abs(ft[:, i]) ** 2 for i in range(len(energies))]
        # zoom to where the density actually lives
        tot = sum(dens_list)
        keep = tot > 1e-4 * tot.max()
        lo, hi = xs[keep][0], xs[keep][-1]
        sel = (xs >= 1.3 * lo) & (xs <= 1.3 * hi)
        xs = xs[sel]; dens_list = [d[sel] for d in dens_list]
        xsym, Vcurve = conj, None
    else:                                   # conjugate of a periodic mode: integer dist.
        _, vecs = _num.eigensystem(result, params, n_levels, cutoffs,
                                   offsets, mode_types)
        ncut = (vecs.shape[0] - 1) // 2
        xs = np.arange(-ncut, ncut + 1)
        dens_list = [np.abs(vecs[:, i]) ** 2 for i in range(len(energies))]
        xsym, Vcurve, discrete = conj, None, True

    if Vcurve is not None:
        ax.plot(xs, Vcurve, color="k", lw=1.5, zorder=5)
    span = ((Vcurve.max() - Vcurve.min()) if Vcurve is not None else
            (energies[-1] - energies[0])) or 1.0
    if scale is None:
        gaps = np.diff(energies)
        scale = 0.7 * (np.median(gaps) if len(gaps) and np.median(gaps) > 0
                       else 0.1 * span)
    for i, dens in enumerate(dens_list):
        dens = dens / (dens.max() or 1.0) * scale
        ax.axhline(energies[i], color=f"C{i}", lw=0.6, ls=":", alpha=0.6)
        if discrete:
            ax.plot(xs, energies[i] + dens, color=f"C{i}", marker="o", ms=3,
                    lw=0.8, label=f"{i}")
        else:
            ax.fill_between(xs, energies[i], energies[i] + dens, color=f"C{i}",
                            alpha=0.5, label=f"{i}")
    ax.set_xlabel(f"${sp.latex(xsym)}$")
    ax.set_ylabel("energy")
    ax.set_title("Potential and eigenstates" if Vcurve is not None
                 else f"Eigenstate densities in {xsym}")
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
        # The potential is a function of one coordinate; the other is the kinetic
        # momentum.  For a Josephson junction that coordinate is the flux
        # (cos phi); for its quantum-phase-slip dual it is the CHARGE (cos q).
        # Pick whichever leaves the potential depending on a single coordinate.
        pos = mom = Vexpr = None
        for cand_pos, cand_mom in ((mode.flux, mode.charge),
                                   (mode.charge, mode.flux)):
            V_try = sp.expand(Hn - Hn.coeff(cand_mom, 2) * cand_mom ** 2)
            if cand_mom not in V_try.free_symbols:
                pos, mom, Vexpr = cand_pos, cand_mom, V_try
                break
        if pos is None:
            raise NotImplementedError(
                "the Hamiltonian couples flux and charge (a gyrator's phi*q "
                "cross term), so it does not split into kinetic + a potential of "
                "a single coordinate -- there is no 1-D potential curve to draw. "
                "Use plot_energy_levels / plot_spectrum; the eigenenergies are "
                "unaffected.")
        c2 = complex(Hn.coeff(mom, 2)).real          # kinetic coefficient
        apos = float(complex(Hn.coeff(pos, 2)).real) or 1.0
        L = 8.0 * (c2 / apos) ** 0.25 if apos > 0 else 12.0
        x = np.linspace(-L, L, grid)
        dx = x[1] - x[0]
        K = _cm_kinetic(grid, dx)                    # approximates -d^2/dx^2
        Vfun = sp.lambdify(pos, Vexpr, "numpy")
        V = np.real(Vfun(x)) * np.ones_like(x)
        Hmat = c2 * K + np.diag(V)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            w, v = np.linalg.eigh(0.5 * (Hmat + Hmat.conj().T))
        n = min(n_levels, len(w))
        return x, V, w[:n], v[:, :n], pos

    # PERIODIC / DUAL_PERIODIC: angle = the compact coordinate
    angle, integer = mode.compact, mode.discrete
    Vexpr = sp.expand(Hn - Hn.coeff(integer, 2) * integer ** 2)
    Vfun = sp.lambdify(angle, Vexpr, "numpy")
    x = np.linspace(-np.pi, np.pi, grid)
    V = np.real(Vfun(x)) * np.ones_like(x)
    _, vecs = _num.eigensystem(result, {str(k): v for k, v in params.items()},
                               n_levels, cutoffs, offsets, mode_types)
    ncut = (vecs.shape[0] - 1) // 2
    ns = np.arange(-ncut, ncut + 1)
    basis = np.exp(1j * np.outer(x, ns)) / np.sqrt(2 * np.pi)   # psi = sum c_n e^{i n x}
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        psis = basis @ vecs
    energies = _num.eigenenergies(result, {str(k): v for k, v in params.items()},
                                  n_levels, cutoffs, offsets, mode_types)
    return x, V, energies, psis, angle


def _cm_kinetic(n, dx):
    """Colbert-Miller DVR matrix approximating ``-d^2/dx^2`` on a uniform grid."""
    import numpy as np
    i = np.arange(n)
    diff = i[:, None] - i[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        K = 2.0 * (-1.0) ** diff / np.where(diff == 0, 1, diff) ** 2
    np.fill_diagonal(K, np.pi ** 2 / 3.0)
    return K / dx ** 2

"""
Interactive spectrum explorer: drag a slider, watch the spectrum move.

This is a *self-contained* explorer built on ``matplotlib.widgets.Slider`` and
fluxcharge's own :meth:`ReductionResult.eigenenergies`, so it works for **every**
circuit fluxcharge can quantize -- including the gyrator and quantum-phase-slip
circuits that ``scqubits`` cannot represent, and the single-mode circuits where
its predefined classes do not apply.  (``scqubits``' own interactive widgets are
Jupyter + ``ipywidgets`` only and limited to the circuits it can represent; for
those, prefer :func:`fluxcharge.interop.cross_check_spectrum`.)

Usage::

    r = library.transmon().hamiltonian(ground="v1")
    spectrum_slider(r, {"E_J": (1, 30), "C": (0.2, 2.0)})

Needs an interactive matplotlib backend (e.g. TkAgg/Qt) for the sliders to be
live; with a headless backend it still builds the figure (testable).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import sympy as sp


def parameter_symbols(result) -> list:
    """Free symbols of the Hamiltonian that are circuit *parameters* -- i.e. not
    the dynamical coordinates (the conjugate flux/charge variables)."""
    coords = set(result.coordinates)
    for a, b, _ in result.conjugate_pairs:
        coords.add(a)
        coords.add(b)
    return sorted((s for s in result.H.free_symbols if s not in coords),
                  key=str)


def _default_ranges(result, span=(0.1, 10.0)) -> "Dict[str, Tuple[float, float, float]]":
    lo, hi = span
    return {str(s): (lo, hi, 1.0) for s in parameter_symbols(result)}


def _normalize_ranges(result, ranges):
    """Coerce *ranges* to ``{name: (lo, hi, init)}``.  Accepts ``(lo, hi)`` or
    ``(lo, hi, init)`` per entry; missing entries default to a generic span."""
    out = _default_ranges(result)
    if ranges:
        for name, spec in ranges.items():
            name = str(name)
            if len(spec) == 2:
                lo, hi = spec
                init = out.get(name, (lo, hi, 0.5 * (lo + hi)))[2]
                init = min(max(init, lo), hi)
            else:
                lo, hi, init = spec
            out[name] = (float(lo), float(hi), float(init))
    return out


def spectrum_levels(result, params, *, n_levels=6, cutoffs=None, relative=True):
    """Eigenenergies for one parameter point (helper around ``eigenenergies``).

    Returns levels measured from the ground state when *relative* (the usual
    spectroscopic convention)."""
    ev = result.eigenenergies(params, n_levels=n_levels,
                              **({"cutoffs": cutoffs} if cutoffs else {}))
    import numpy as np
    ev = np.asarray(ev, dtype=float)
    return ev - ev[0] if relative and len(ev) else ev


def spectrum_slider(result, ranges: Optional[Dict] = None, *, n_levels: int = 6,
                    cutoffs: Optional[Dict] = None, relative: bool = True,
                    title: Optional[str] = None, show: bool = True):
    """Open an interactive window: energy levels (rows) vs. parameter sliders.

    * ``result`` -- a :class:`~fluxcharge.reduction.ReductionResult`.
    * ``ranges`` -- ``{param: (lo, hi)}`` or ``{param: (lo, hi, init)}``; any
      parameter omitted gets a generic span.  Defaults to all H parameters.
    * ``relative`` -- plot ``E_i - E_0`` (default) or absolute energies.

    Returns ``(fig, sliders)``; in a headless backend nothing is shown but the
    figure is fully built (and one update has run), which is what the tests check.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    rng = _normalize_ranges(result, ranges)
    names = list(rng)
    if not names:
        raise ValueError("the Hamiltonian has no free parameters to vary "
                         f"(H = {result.H}); nothing to slide.")

    init = {n: rng[n][2] for n in names}
    levels0 = spectrum_levels(result, init, n_levels=n_levels,
                              cutoffs=cutoffs, relative=relative)

    fig = plt.figure(figsize=(7.5, 5.0))
    n_sl = len(names)
    ax = fig.add_axes([0.12, 0.18 + 0.06 * n_sl, 0.82, 0.74 - 0.06 * n_sl])
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_ylabel(r"$E_i - E_0$" if relative else r"$E_i$")
    ax.set_title(title or getattr(result, "title", None) or "spectrum")

    lines = [ax.axhline(float(y), color=f"C{i % 10}", lw=2) for i, y in enumerate(levels0)]
    labels = [ax.text(1.005, float(y), f"$|{i}\\rangle$", va="center", fontsize=9)
              for i, y in enumerate(levels0)]
    ax.set_ylim(float(min(levels0)) - 0.5, float(max(levels0)) * 1.05 + 0.5)

    sliders = {}
    for k, name in enumerate(names):
        lo, hi, ini = rng[name]
        sax = fig.add_axes([0.12, 0.06 + 0.05 * k, 0.78, 0.03])
        sliders[name] = Slider(sax, f"${sp.latex(sp.Symbol(name))}$", lo, hi, valinit=ini)

    def update(_event=None):
        params = {n: sliders[n].val for n in names}
        try:
            lv = spectrum_levels(result, params, n_levels=n_levels,
                                 cutoffs=cutoffs, relative=relative)
        except Exception as exc:  # keep the window alive on a bad point
            ax.set_title(f"error: {type(exc).__name__}")
            fig.canvas.draw_idle()
            return
        for i, ln in enumerate(lines):
            y = float(lv[i]) if i < len(lv) else np.nan
            ln.set_ydata([y, y])
            labels[i].set_y(y)
        finite = lv[np.isfinite(lv)]
        if len(finite):
            ax.set_ylim(float(finite.min()) - 0.5, float(finite.max()) * 1.05 + 0.5)
        ax.set_title(title or getattr(result, "title", None) or "spectrum")
        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(update)
    update()  # ensure a consistent first frame (also exercises the path headless)

    if show:
        plt.show()
    return fig, sliders

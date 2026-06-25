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


def is_flux_bias(name: str) -> bool:
    """External-flux parameter (``phi_ext_<loop>``), period ``2*pi``."""
    return str(name).startswith("phi_ext_")


def is_charge_bias(name: str) -> bool:
    """Offset/gate-charge parameter (``n_g_<node>``), period ``1``."""
    return str(name).startswith("n_g_")


def _default_ranges(result, span=(0.1, 10.0)) -> "Dict[str, Tuple[float, float, float]]":
    """Per-parameter ``(lo, hi, init)`` defaults.

    External biases get physical spans over one period -- an external flux runs
    ``0..2*pi`` (sweet spot at ``pi``), an offset charge ``0..1`` Cooper pair --
    so the slider sweeps exactly the periodic structure (flux modulation, charge
    dispersion).  Other parameters (``C``, ``L``, ``E_J`` ...) get a generic span.
    """
    import math
    lo, hi = span
    out = {}
    for s in parameter_symbols(result):
        name = str(s)
        if is_flux_bias(name):
            out[name] = (0.0, 2 * math.pi, math.pi)
        elif is_charge_bias(name):
            out[name] = (0.0, 1.0, 0.0)
        else:
            out[name] = (lo, hi, 1.0)
    return out


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


def _auto_sweep(result, names):
    """Pick a default x-axis parameter: prefer an external bias (the physically
    interesting sweep -- flux modulation / charge dispersion), else the first."""
    for pred in (is_flux_bias, is_charge_bias):
        for n in names:
            if pred(n):
                return n
    return names[0]


def _levels_to_quantity(levels, quantity):
    """``levels`` (E_i, ground-referenced) -> the plotted curves.

    * ``"levels"``      -- E_i - E_0 (one curve per level).
    * ``"transitions"`` -- consecutive gaps f_{i,i+1} = E_{i+1} - E_i.
    """
    import numpy as np
    levels = np.asarray(levels, dtype=float)
    if quantity == "transitions":
        return np.diff(levels)
    return levels


def spectrum_vs_param(result, sweep: Optional[str] = None, ranges: Optional[Dict] = None,
                      *, n_levels: int = 6, cutoffs: Optional[Dict] = None,
                      quantity: str = "levels", npoints: int = 41,
                      relative: bool = True, title: Optional[str] = None,
                      show: bool = True):
    """Plot the spectrum as **curves vs one swept parameter**, with sliders for
    the rest (the ``scqubits.plot_evals_vs_paramvals`` pattern, made live).

    * ``sweep`` -- the x-axis parameter; defaults to an external bias if present
      (flux/charge sweep), else the first parameter.
    * ``quantity`` -- ``"levels"`` (E_i - E_0) or ``"transitions"`` (consecutive
      gaps f_{i,i+1}, e.g. f01, f12 -- spectroscopy lines).
    * the swept parameter shows a movable marker; dragging another slider
      recomputes every curve.

    Returns ``(fig, sliders)``.  Built fully in a headless backend (testable).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    if quantity not in ("levels", "transitions"):
        raise ValueError("quantity must be 'levels' or 'transitions'")

    rng = _normalize_ranges(result, ranges)
    names = list(rng)
    if not names:
        raise ValueError(f"the Hamiltonian has no free parameters (H = {result.H}).")
    sweep = str(sweep) if sweep is not None else _auto_sweep(result, names)
    if sweep not in rng:
        raise ValueError(f"sweep parameter {sweep!r} is not among {names}")
    others = [n for n in names if n != sweep]

    lo, hi, init = rng[sweep]
    xs = np.linspace(lo, hi, npoints)

    n_curves = n_levels - 1 if quantity == "transitions" else n_levels
    curve_labels = ([f"$f_{{{i}{i+1}}}$" for i in range(n_curves)] if quantity == "transitions"
                    else [f"$|{i}\\rangle$" for i in range(n_curves)])

    def compute(slider_vals):
        Y = np.full((len(xs), n_curves), np.nan)
        for j, x in enumerate(xs):
            params = dict(slider_vals); params[sweep] = float(x)
            try:
                lv = spectrum_levels(result, params, n_levels=n_levels,
                                     cutoffs=cutoffs, relative=relative)
                q = _levels_to_quantity(lv, quantity)
                Y[j, :len(q)] = q[:n_curves]
            except Exception:
                pass  # leave NaN at unreachable points
        return Y

    fig = plt.figure(figsize=(7.5, 5.2))
    n_sl = len(others)
    ax = fig.add_axes([0.12, 0.16 + 0.06 * n_sl, 0.82, 0.76 - 0.06 * n_sl])
    xlabel = sweep
    if is_flux_bias(sweep):
        xlabel = r"$\Phi_\mathrm{ext}/\varphi_0$  (" + sweep + ")"
    elif is_charge_bias(sweep):
        xlabel = r"$n_g$  (" + sweep + ")"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("transition (GHz)" if quantity == "transitions"
                  else (r"$E_i - E_0$" if relative else r"$E_i$"))
    ax.set_title(title or getattr(result, "title", None) or "spectrum vs " + sweep)

    Y0 = compute({n: rng[n][2] for n in others})
    curves = [ax.plot(xs, Y0[:, i], color=f"C{i % 10}", lw=1.8, label=curve_labels[i])[0]
              for i in range(n_curves)]
    marker = ax.axvline(init, color="0.4", ls="--", lw=1)
    ax.set_xlim(lo, hi)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    sliders = {}
    # a slider for the sweep value (moves the marker) plus one per other param
    sax0 = fig.add_axes([0.12, 0.06 + 0.05 * n_sl, 0.78, 0.03])
    sliders[sweep] = Slider(sax0, f"${sp.latex(sp.Symbol(sweep))}$", lo, hi, valinit=init)
    for k, name in enumerate(others):
        a, b, ini = rng[name]
        sax = fig.add_axes([0.12, 0.06 + 0.05 * k, 0.78, 0.03])
        sliders[name] = Slider(sax, f"${sp.latex(sp.Symbol(name))}$", a, b, valinit=ini)

    def _rescale(Y):
        finite = Y[np.isfinite(Y)]
        if len(finite):
            pad = 0.05 * (finite.max() - finite.min() + 1e-9)
            ax.set_ylim(float(finite.min()) - pad, float(finite.max()) + pad)

    def update(_=None):
        Y = compute({n: sliders[n].val for n in others})
        for i, ln in enumerate(curves):
            ln.set_ydata(Y[:, i])
        marker.set_xdata([sliders[sweep].val, sliders[sweep].val])
        _rescale(Y)
        fig.canvas.draw_idle()

    for n, s in sliders.items():
        # moving the sweep slider only repositions the marker (cheap); the other
        # sliders trigger a full recompute of the curves
        s.on_changed((lambda _=None: (marker.set_xdata([sliders[sweep].val] * 2),
                                       fig.canvas.draw_idle())) if n == sweep else update)
    _rescale(Y0)

    if show:
        plt.show()
    return fig, sliders

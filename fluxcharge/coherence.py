"""
Matrix elements and coherence estimates for a reduced flux-charge circuit.

This builds on :mod:`fluxcharge.numerics`: it reuses the same operator
construction and eigensystem, so operators and eigenvectors live in the same
basis.  The pieces here are deliberately split into an *exact* core and
*model-dependent* estimates, and the latter are explicit about their
assumptions (this package would rather surface a convention than hide it):

* **Exact** -- transition matrix elements ``<i|O|j>`` of any coordinate
  operator (charge ``n`` or flux ``phi``), and the bias *sensitivity*
  ``df_ij/d(lambda)`` of a transition frequency to an external parameter.  The
  sensitivity is the convention-free heart of dephasing: it is zero exactly at a
  sweet spot.

* **Model-dependent** -- a depolarization rate ``1/T1`` from Fermi's golden rule
  given a noise operator and a spectral density ``S(omega)``, and a ``1/f``
  pure-dephasing rate from the bias sensitivity and a noise amplitude.  The
  numbers are only as good as the supplied ``S(omega)`` / amplitudes; defaults
  are typical superconducting-qubit values, not ground truth.

All energies/rates are in the unit the Hamiltonian parameters were given in --
use :func:`fluxcharge.units.to_natural` so they come out in GHz.
"""

from __future__ import annotations

import sympy as sp


def _np():
    import numpy as np
    return np


def _solve(result, params, n_levels, cutoffs, offsets, mode_types):
    """One operator builder + eigensystem (operators & eigvecs share a basis)."""
    from .numerics import _OperatorBuilder
    np = _np()
    result = result if result.is_canonical else result.canonical()
    builder = _OperatorBuilder(result, params, cutoffs, offsets, mode_types)
    H = builder.matrix()
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        w, v = np.linalg.eigh(H)
    n = min(n_levels, len(w))
    return builder, w[:n], v[:, :n]


def operator_matrix(result, coordinate, params=None, cutoffs=None,
                    offsets=None, mode_types=None):
    """Full-space matrix of a coordinate operator (a charge ``q_*`` or an
    ``EXTENDED`` flux ``phi_*``).  Raises for a compact flux, which has no bare
    operator (only ``cos``)."""
    builder, _, _ = _solve(result, params, 1, cutoffs, offsets, mode_types)
    sym = sp.sympify(coordinate)
    if sym not in builder.op:
        raise ValueError(
            f"{sym} has no bare operator (a compact/periodic flux only appears "
            f"inside a cosine); available: {sorted(map(str, builder.op))}")
    return builder.op[sym]


def matrix_elements(result, coordinate, params=None, n_levels=6, cutoffs=None,
                    offsets=None, mode_types=None):
    """The ``n_levels x n_levels`` matrix of ``<i|coordinate|j>`` (complex).

    *coordinate* is a charge (``"q_f1"``) or extended flux (``"phi_v2"``) symbol.
    """
    np = _np()
    builder, _, v = _solve(result, params, n_levels, cutoffs, offsets, mode_types)
    sym = sp.sympify(coordinate)
    if sym not in builder.op:
        raise ValueError(
            f"{sym} has no bare operator; available: {sorted(map(str, builder.op))}")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return v.conj().T @ builder.op[sym] @ v


def transition_sensitivity(result, bias, params, levels=(0, 1), delta=1e-4,
                           n_levels=None, cutoffs=None, offsets=None,
                           mode_types=None):
    """First and second derivatives of a transition frequency ``f_ij`` w.r.t. an
    external parameter *bias* (a symbol/name in ``params``), by central finite
    difference.

    Returns ``(df, d2f)``.  ``df = 0`` marks a first-order-insensitive sweet
    spot (e.g. a flux qubit at half flux, a transmon at integer offset charge).
    """
    from .numerics import eigenenergies
    i, j = levels
    n = n_levels or (max(i, j) + 1)
    bias = str(sp.sympify(bias))
    x0 = float(params[bias])

    def f(x):
        p = dict(params); p[bias] = x
        ev = eigenenergies(result, p, n_levels=n, cutoffs=cutoffs,
                           offsets=offsets, mode_types=mode_types)
        return ev[j] - ev[i]

    fp, f0, fm = f(x0 + delta), f(x0), f(x0 - delta)
    df = (fp - fm) / (2 * delta)
    d2f = (fp - 2 * f0 + fm) / delta ** 2
    return df, d2f


def t1(result, params, noise_operator, spectral_density, levels=(0, 1),
       n_levels=None, cutoffs=None, offsets=None, mode_types=None):
    """Depolarization time ``T1`` from Fermi's golden rule.

    ``1/T1 = |<i|A|j>|**2 * [S(w_ij) + S(-w_ij)]`` where ``A`` is the
    *noise_operator* (a coordinate symbol, e.g. the charge for dielectric loss
    or the flux for inductive loss) and ``spectral_density`` is a callable
    ``S(omega)`` (omega in the same angular-frequency unit as the spectrum).
    Returns ``(T1, rate)``.  This is exact given ``A`` and ``S``; the physics
    lives entirely in the spectral density you pass.
    """
    np = _np()
    i, j = levels
    n = n_levels or (max(i, j) + 1)
    builder, w, v = _solve(result, params, n, cutoffs, offsets, mode_types)
    sym = sp.sympify(noise_operator)
    if sym not in builder.op:
        raise ValueError(f"{sym} has no bare operator for the noise coupling")
    A = builder.op[sym]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        mel = v[:, i].conj() @ A @ v[:, j]
    omega = 2 * np.pi * (w[j] - w[i])             # angular frequency
    rate = np.abs(mel) ** 2 * (spectral_density(omega) + spectral_density(-omega))
    rate = float(np.real(rate))
    return (np.inf if rate == 0 else 1.0 / rate), rate


def dephasing_1_over_f(result, bias, params, noise_amplitude, levels=(0, 1),
                       t_exp=10e-6, omega_ir=2 * 3.141592653589793 * 1.0,
                       freq_to_hz=1e9, **kw):
    """Pure-dephasing rate ``1/Tphi`` from ``1/f`` noise in an external *bias*.

    First order (away from a sweet spot)::

        Gamma_phi = |df_ij/d(bias)| * A * sqrt(2 |ln(omega_ir * t_exp)|),

    with *noise_amplitude* ``A`` the ``1/f`` amplitude at 1 Hz in the bias's own
    units (e.g. ~1e-6 * 2*pi for flux in rad, ~1e-4 for offset charge).  Near a
    sweet spot the first-order term vanishes and the second-order term in
    ``d2f`` dominates; both are returned.  *freq_to_hz* converts the spectrum's
    frequency unit to Hz for the rate (1e9 if the spectrum is in GHz).

    Returns a dict with ``df``, ``d2f``, ``gamma_phi`` [1/s], ``t_phi`` [s].
    """
    np = _np()
    df, d2f = transition_sensitivity(result, bias, params, levels=levels, **kw)
    ln = np.sqrt(2 * abs(np.log(omega_ir * t_exp)))
    gamma1 = abs(df) * freq_to_hz * noise_amplitude * ln
    gamma2 = abs(d2f) * freq_to_hz * noise_amplitude ** 2 * ln ** 2   # crude 2nd order
    gamma = gamma1 if gamma1 > 0 else gamma2
    return {"df": df, "d2f": d2f, "gamma_phi": gamma,
            "t_phi": (np.inf if gamma == 0 else 1.0 / gamma)}

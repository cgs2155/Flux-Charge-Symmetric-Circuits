"""
Physical units for the (otherwise dimensionless) flux-charge formalism.

The package works in natural units (``hbar = 1``, reduced flux quantum ``= 1``,
``2e = 1``), so a reduced Hamiltonian and its eigenvalues are dimensionless.
Experimenters think in **fF / nH / GHz**, so this module converts a circuit's
physical element values into the numeric parameters the solver expects, chosen
so that **the resulting eigenenergies come out in GHz** (frequency units, i.e.
energy / h).

Conventions (the standard superconducting-qubit ones):

* charging energy of a capacitor   ``E_C = e**2 / (2C)``        ->  ``19.37 GHz / C[fF]``
* inductive energy of an inductor  ``E_L = (Phi0/2pi)**2 / L``  ->  ``163.5 GHz / L[nH]``
* Josephson / phase-slip energies  ``E_J``, ``E_S``            given directly in GHz

In the package's symbols a capacitor contributes ``q**2/(2C)`` and an inductor
``phi**2/(2L)``; matching these to ``4 E_C n**2`` and ``E_L phi**2 / 2`` fixes
the natural-unit value of each symbol: ``C -> 1/(8 E_C)``, ``L -> 1/E_L`` (with
``E_C``, ``E_L`` in GHz), while ``E_J``/``E_S`` pass straight through and
dimensionless quantities (gyration ratio, external flux in rad, offset charge)
are used as given.

Example::

    ckt = transmon()                       # symbols C, E_J
    p = to_natural(ckt, {"C": "70fF", "E_J": "15GHz"})
    ckt.hamiltonian(ground="v1").eigenenergies(p)      # -> eigenvalues in GHz
"""

from __future__ import annotations

import re

# CODATA 2019 exact constants
_E = 1.602176634e-19          # elementary charge [C]
_H = 6.62607015e-34           # Planck constant [J s]
_PHI0 = _H / (2 * _E)         # flux quantum [Wb]
_PHI0_RED = _PHI0 / (2 * 3.141592653589793)   # reduced flux quantum [Wb]

# E_C[GHz] = K_C / C[fF];  E_L[GHz] = K_L / L[nH]
K_C = (_E ** 2 / (2 * _H)) / 1e-15 / 1e9     # GHz * fF  (~19.37)
K_L = (_PHI0_RED ** 2 / _H) / 1e-9 / 1e9     # GHz * nH  (~163.5)

_UNIT_SCALE = {            # multiply to get the SI base used above
    # capacitance -> fF
    "ff": ("C", 1.0), "pf": ("C", 1e3), "nf": ("C", 1e6), "f": ("C", 1e15),
    # inductance -> nH
    "nh": ("L", 1.0), "uh": ("L", 1e3), "µh": ("L", 1e3), "ph": ("L", 1e-3),
    "h": ("L", 1e9), "mh": ("L", 1e6),
    # frequency (energy/h) -> GHz
    "ghz": ("F", 1.0), "mhz": ("F", 1e-3), "khz": ("F", 1e-6), "hz": ("F", 1e-9),
}


def parse_quantity(q):
    """Parse ``"70fF"`` / ``(70, "fF")`` / ``0.27`` into ``(value, kind)``.

    *kind* is ``"C"`` (capacitance, fF), ``"L"`` (inductance, nH), ``"F"``
    (frequency/energy, GHz) or ``None`` (a bare number, dimensionless or already
    in the target unit).
    """
    if isinstance(q, (int, float)):
        return float(q), None
    if isinstance(q, (tuple, list)) and len(q) == 2:
        value, unit = float(q[0]), str(q[1])
    else:
        m = re.fullmatch(r"\s*([-+0-9.eE]+)\s*([a-zA-Zµ]*)\s*", str(q))
        if not m:
            raise ValueError(f"cannot parse quantity {q!r}")
        value = float(m.group(1))
        unit = m.group(2)
    if not unit:
        return value, None
    key = unit.lower()
    if key not in _UNIT_SCALE:
        raise ValueError(f"unknown unit {unit!r} in {q!r}")
    kind, scale = _UNIT_SCALE[key]
    return value * scale, kind


def _role_map(circuit):
    """Map each parameter symbol name to its role: 'C', 'L', 'E' or '' (free)."""
    from .elements import (Capacitor, Inductor, JosephsonJunction,
                           QuantumPhaseSlip, Gyrator)
    roles = {}
    for el in circuit._elements:
        if isinstance(el, Capacitor):
            roles[str(el.C)] = "C"
        elif isinstance(el, Inductor):
            roles[str(el.L)] = "L"
        elif isinstance(el, JosephsonJunction):
            roles[str(el.EJ)] = "E"
        elif isinstance(el, QuantumPhaseSlip):
            roles[str(el.ES)] = "E"
        elif isinstance(el, Gyrator):
            roles.setdefault(str(el.G), "")
    for v in list(getattr(circuit, "_flux_bias", {}).values()) \
            + list(getattr(circuit, "_offset_charge", {}).values()):
        for s in v.free_symbols:
            roles.setdefault(str(s), "")
    return roles


def charging_energy_GHz(C_fF):
    """Charging energy ``E_C = e**2/2C`` in GHz from a capacitance in fF."""
    return K_C / C_fF


def inductive_energy_GHz(L_nH):
    """Inductive energy ``E_L = (Phi0/2pi)**2/L`` in GHz from an inductance in nH."""
    return K_L / L_nH


def to_natural(circuit, physical):
    """Numeric parameters (natural units) for *circuit* from *physical* values.

    *physical* maps a parameter-symbol name to a value with units, e.g.
    ``{"C": "70fF", "L": "150nH", "E_J": "15GHz", "G": 0.5}``.  Capacitances may
    be given as a capacitance (``fF``) or directly as a charging energy
    (``GHz``); inductances as an inductance (``nH``) or an inductive energy
    (``GHz``); Josephson/phase-slip values as energies (``GHz``).  The returned
    dict, passed to ``eigenenergies`` / ``sweep`` / ``hamiltonian_matrix``,
    yields **eigenvalues in GHz**.
    """
    roles = _role_map(circuit)
    out = {}
    for name, q in physical.items():
        value, kind = parse_quantity(q)
        role = roles.get(name, "")
        if role == "C":
            E_C = charging_energy_GHz(value) if kind == "C" else value
            out[name] = 1.0 / (8.0 * E_C)
        elif role == "L":
            E_L = inductive_energy_GHz(value) if kind == "L" else value
            out[name] = 1.0 / E_L
        else:                       # energy (E_J/E_S, given in GHz) or dimensionless
            out[name] = value
    return out

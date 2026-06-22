"""
Import circuits written in scqubits' ``Circuit`` YAML format.

scqubits describes a circuit as a list of branches::

    branches:
    - [JJ, 1, 0, EJ = 10, 20]      # Josephson energy EJ and junction charging energy EC
    - [L,  1, 0, 0.5]              # inductive energy EL
    - [C,  1, 0, 2.5]              # charging energy EC

Branch parameters are *energies* (GHz): ``EC = e**2/2C`` for a capacitor,
``EL = (Phi0/2pi)**2/L`` for an inductor, ``EJ`` for a junction.  Values may
also carry scqubits' unit suffixes -- a capacitance ``EC = 90 fF``, an
inductance ``EL = 5 nH``, or an explicit ``EJ = 15 GHz`` -- which are converted
to the corresponding energy.  Node ``0`` is ground.  This maps cleanly onto fluxcharge (which shares the ``2e = 1``
convention, so ``EC = 1/(8C)``, ``EL = 1/L``): each branch energy ``X`` named in
the YAML becomes a fluxcharge parameter ``X`` and the element value is set so the
Hamiltonian matches.  A ``JJ`` branch becomes a junction **plus** a parallel
capacitor for its junction charging energy.

:func:`from_scqubits_yaml` returns ``(circuit, params)``: a fluxcharge
:class:`~fluxcharge.circuit.Circuit` (loops auto-inferred, so nothing else is
needed) and a dict of the numeric parameter values from the YAML, ready for
``eigenenergies`` -- giving a spectrum in the same (GHz) units as scqubits.

Only reciprocal ``C`` / ``L`` / ``JJ`` branches exist in scqubits; once imported
you can of course add gyrators and quantum phase slips, which scqubits cannot
represent.
"""

from __future__ import annotations

import os

import sympy as sp

from .circuit import Circuit
from .units import parse_quantity, charging_energy_GHz, inductive_energy_GHz


def _energy(tok, params):
    """Resolve one branch parameter to an **energy in GHz** (sympy value).

    Accepts ``"EJ = 10"`` (named, GHz energy), a bare number (``"20"``, GHz), a
    bare symbol (``"EJ"``, a free parameter), and -- the point of this -- values
    carrying scqubits unit suffixes: a capacitance (``"EC = 2 fF"``) is converted
    to its charging energy ``e**2/2C``, an inductance (``"EL = 5 nH"``) to its
    inductive energy, and an explicit ``GHz``/``Hz``/``J`` value is used as the
    energy directly.  A numeric default is recorded in *params* under the symbol
    name so the returned circuit diagonalizes (in GHz) out of the box.
    """
    tok = tok.strip()
    if "=" in tok:
        name, rhs = (x.strip() for x in tok.split("=", 1))
    else:
        name, rhs = None, tok
    try:
        value, kind = parse_quantity(rhs)        # (number, 'C'|'L'|'F'|None)
    except ValueError:
        return sp.Symbol(name or rhs)            # a bare symbolic parameter
    if kind == "C":                              # a capacitance -> charging energy
        energy = charging_energy_GHz(value)
    elif kind == "L":                            # an inductance -> inductive energy
        energy = inductive_energy_GHz(value)
    else:                                        # already an energy (GHz) or bare
        energy = value
    if name is None:
        return sp.Float(energy)
    params[name] = energy
    return sp.Symbol(name)


def _branches(text):
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.lower().startswith("branches"):
            continue
        inner = line.lstrip("-").strip()
        if not (inner.startswith("[") and inner.endswith("]")):
            continue
        toks = [t.strip().strip('"').strip("'") for t in inner[1:-1].split(",")]
        if len(toks) >= 3:
            yield toks


def from_scqubits_yaml(source):
    """Build a fluxcharge ``Circuit`` from an scqubits ``Circuit`` YAML string or
    file.  Returns ``(circuit, params)``; diagonalize with
    ``circuit.hamiltonian(...).eigenenergies(params)`` for a GHz spectrum.
    """
    if os.path.exists(source) or source.endswith((".yaml", ".yml")):
        with open(source) as fh:
            text = fh.read()
    else:
        text = source

    ckt = Circuit()
    ckt.title = "imported from scqubits"
    params = {}
    nodes = set()

    for i, toks in enumerate(_branches(text), start=1):
        btype, n1, n2 = toks[0].upper(), toks[1], toks[2]
        bparams = toks[3:]
        nodes.update((n1, n2))
        if btype == "C":
            EC = _energy(bparams[0], params)
            ckt.add_capacitor(f"e{i}", n1, n2, C=1 / (8 * EC))
        elif btype == "L":
            EL = _energy(bparams[0], params)
            ckt.add_inductor(f"e{i}", n1, n2, L=1 / EL)
        elif btype == "JJ":
            EJ = _energy(bparams[0], params)
            ckt.add_josephson(f"e{i}", n1, n2, EJ=EJ)
            if len(bparams) >= 2:                       # junction capacitance
                EC = _energy(bparams[1], params)
                ckt.add_capacitor(f"eC{i}", n1, n2, C=1 / (8 * EC))
        else:
            raise NotImplementedError(
                f"scqubits branch type {toks[0]!r} is not supported (only C, L, JJ; "
                "JJ harmonics / JJs arrays / mutual inductance ML have no single "
                "flux-charge element). Model it explicitly if needed.")

    ckt.ground = "0" if "0" in nodes else None
    return ckt, params

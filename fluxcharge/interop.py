"""
Import circuits written in scqubits' ``Circuit`` YAML format.

scqubits describes a circuit as a list of branches::

    branches:
    - [JJ, 1, 0, EJ = 10, 20]      # Josephson energy EJ and junction charging energy EC
    - [L,  1, 0, 0.5]              # inductive energy EL
    - [C,  1, 0, 2.5]              # charging energy EC

Branch parameters are *energies* (GHz): ``EC = e**2/2C`` for a capacitor,
``EL = (Phi0/2pi)**2/L`` for an inductor, ``EJ`` for a junction.  Node ``0`` is
ground.  This maps cleanly onto fluxcharge (which shares the ``2e = 1``
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


def _is_float(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _parse_param(tok):
    """``"EJ = 10"`` -> ``("EJ", 10.0)``; ``"20"`` -> ``(None, 20.0)``;
    ``"EJ"`` -> ``("EJ", None)``."""
    tok = tok.strip()
    if "=" in tok:
        name, val = (x.strip() for x in tok.split("=", 1))
        return name, (float(val) if _is_float(val) else None)
    if _is_float(tok):
        return None, float(tok)
    return tok, None


def _resolve(tok, params):
    """Return the sympy value for an element parameter, recording any numeric
    default in *params* (keyed by the symbol name)."""
    name, num = _parse_param(tok)
    if name is None:                      # a bare number
        return sp.Float(num)
    if num is not None:
        params[name] = num
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
            EC = _resolve(bparams[0], params)
            ckt.add_capacitor(f"e{i}", n1, n2, C=1 / (8 * EC))
        elif btype == "L":
            EL = _resolve(bparams[0], params)
            ckt.add_inductor(f"e{i}", n1, n2, L=1 / EL)
        elif btype == "JJ":
            EJ = _resolve(bparams[0], params)
            ckt.add_josephson(f"e{i}", n1, n2, EJ=EJ)
            if len(bparams) >= 2:                       # junction capacitance
                EC = _resolve(bparams[1], params)
                ckt.add_capacitor(f"eC{i}", n1, n2, C=1 / (8 * EC))
        else:
            raise NotImplementedError(
                f"scqubits branch type {toks[0]!r} is not supported (only C, L, JJ; "
                "JJ harmonics / JJs arrays / mutual inductance ML have no single "
                "flux-charge element). Model it explicitly if needed.")

    ckt.ground = "0" if "0" in nodes else None
    return ckt, params

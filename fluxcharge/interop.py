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


# ----------------------------------------------------------------------
# export + cross-validation
# ----------------------------------------------------------------------
def _num(expr, params):
    """Numeric float value of a (possibly symbolic) element value."""
    val = sp.sympify(expr).subs(params or {})
    val = sp.N(val)
    if not val.is_number:
        free = ", ".join(sorted(str(s) for s in val.free_symbols))
        raise ValueError(f"cannot export: value {expr} still depends on {free}; "
                         "pass numeric params for every symbol")
    return float(val)


def to_scqubits_yaml(circuit, params=None):
    """Serialise a **reciprocal** fluxcharge circuit to scqubits ``Circuit`` YAML.

    Capacitors become ``C`` branches (charging energy ``EC = 1/(8 C)``),
    inductors ``L`` branches (``EL = 1/L``), and each Josephson junction a ``JJ``
    branch ``[JJ, i, j, EJ, EC_J]`` whose junction charging energy is taken from
    the capacitor in parallel with it (every physical junction has one; that
    capacitor is then not emitted separately).  Node ``0`` is ground.

    Gyrators and quantum phase slips are refused -- scqubits has no
    representation for a non-reciprocal element or a cosine-of-charge.  All
    element values must be numeric after substituting *params*.

    Returns the YAML string; feed it to ``scqubits.Circuit(yaml, from_file=False)``.
    """
    from .elements import Capacitor, Inductor, JosephsonJunction

    bad = [type(el).__name__ for el in circuit._elements
           if type(el).__name__ not in ("Capacitor", "Inductor", "JosephsonJunction")]
    if bad:
        raise NotImplementedError(
            f"scqubits cannot represent {sorted(set(bad))}; export is limited to "
            "the reciprocal C / L / JJ subset (no gyrators or quantum phase slips)")

    # node -> integer, ground = 0
    ground = getattr(circuit, "ground", None)
    others = [v for v in circuit.vertices if v != ground]
    nmap = {ground: 0} if ground is not None else {}
    for k, v in enumerate(others, start=1 if ground is not None else 0):
        nmap[v] = k

    def pair(el):
        return nmap[el._edge.tail], nmap[el._edge.head]

    # match each JJ to a parallel capacitor (same unordered node pair)
    caps = [el for el in circuit._elements if isinstance(el, Capacitor)]
    consumed = set()
    lines = ["branches:"]
    for el in circuit._elements:
        if not isinstance(el, JosephsonJunction):
            continue
        i, j = pair(el)
        partner = next((c for c in caps if id(c) not in consumed
                        and {nmap[c._edge.tail], nmap[c._edge.head]} == {i, j}), None)
        if partner is None:
            raise NotImplementedError(
                f"junction {el.name} has no parallel capacitor; scqubits requires a "
                "junction charging energy. Add the junction capacitance explicitly.")
        consumed.add(id(partner))
        EJ = _num(el.EJ, params)
        EC = 1.0 / (8.0 * _num(partner.C, params))
        lines.append(f"- [JJ, {i}, {j}, {EJ:.12g}, {EC:.12g}]")

    for el in circuit._elements:
        if isinstance(el, Capacitor) and id(el) not in consumed:
            i, j = pair(el)
            lines.append(f"- [C, {i}, {j}, {1.0/(8.0*_num(el.C, params)):.12g}]")
        elif isinstance(el, Inductor):
            i, j = pair(el)
            lines.append(f"- [L, {i}, {j}, {1.0/_num(el.L, params):.12g}]")
    return "\n".join(lines) + "\n"


def cross_check_spectrum(circuit, params, n_levels=5, ground=None,
                         open_loops=None, cutoffs=None, scqubits_cutoff=None,
                         **hamiltonian_kw):
    """Diagonalise *circuit* with both fluxcharge and scqubits and compare.

    Exports the (reciprocal) circuit to scqubits, diagonalises it there, runs
    fluxcharge's own numeric diagonalisation, and returns a dict with both
    ground-referenced spectra (GHz) and ``max_abs_diff``.

    CAVEAT -- which scqubits is the oracle.  scqubits' *general* ``Circuit``
    class is a clean reference only for charge-network circuits (no linear
    inductor): the transmon round-trips to ~1e-13.  For inductive circuits its
    general ``Circuit`` class uses a convention that disagrees with both an
    independent grid diagonalisation and scqubits' own predefined classes
    (e.g. ``Fluxonium``) -- fluxcharge matches the grid and the predefined
    classes, ``Circuit`` is the outlier.  So a nonzero ``max_abs_diff`` against
    ``Circuit`` on an inductive circuit may be scqubits' quirk, not fluxcharge's
    -- cross-check against a grid or a predefined class before concluding.  Use
    this harness as a clean oracle for charge networks; for inductive/multi-mode
    circuits prefer the grid references in the test suite.
    """
    import numpy as np
    import scqubits as scq

    yaml = to_scqubits_yaml(circuit, params)
    scq_circ = scq.Circuit(yaml, from_file=False)
    if scqubits_cutoff:
        for sym in scq_circ.cutoff_names:
            setattr(scq_circ, sym, scqubits_cutoff)
    ev_sc = np.sort(scq_circ.eigenvals(evals_count=n_levels))
    ev_sc = ev_sc - ev_sc[0]

    res = circuit.hamiltonian(ground=ground, open_loops=open_loops,
                              **hamiltonian_kw)
    ev_fc = np.sort(res.eigenenergies(params, n_levels=n_levels, cutoffs=cutoffs))
    ev_fc = ev_fc - ev_fc[0]

    return {
        "scqubits": ev_sc,
        "fluxcharge": ev_fc,
        "max_abs_diff": float(np.max(np.abs(ev_sc - ev_fc))),
        "scqubits_yaml": yaml,
    }

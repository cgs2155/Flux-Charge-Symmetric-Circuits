"""
Validation tests for :mod:`fluxcharge`.

These check the package against the manuscript's worked example (whose
Lagrangian and Hamiltonian are known in closed form), a pure-null-vector
circuit, and a set of textbook circuits -- the LC oscillator, transmon,
fluxonium and a quantum-phase-slip circuit.  They can be run with ``pytest`` or
directly with ``python``.
"""

import sympy as sp

from fluxcharge import Circuit

try:
    import pytest
except ModuleNotFoundError:  # allow running directly without pytest installed
    import contextlib

    class _PytestShim:
        @staticmethod
        @contextlib.contextmanager
        def raises(exc):
            try:
                yield
            except exc:
                return
            raise AssertionError(f"expected {exc.__name__} to be raised")

    pytest = _PytestShim()


def _circulator():
    ckt = Circuit()
    ckt.add_josephson("e1", "v1", "v2", EJ="E_J")
    ckt.add_capacitor("e2", "v2", "v3", C="C")
    ckt.add_capacitor("e3", "v3", "v1", C="C")
    ckt.add_gyrator(("e4", "v1", "v3"), ("e5", "v2", "v3"), G="G")
    ckt.add_loop("f1", ["+e3", "+e4"])
    ckt.add_loop("f2", ["+e1", "-e4", "+e5"])
    ckt.add_loop("f3", ["+e2", "-e5"])
    ckt.add_loop("f4", ["-e1", "-e2", "-e3"])
    return ckt


def test_exactness():
    """Kirchhoff's laws:  B * A = 0."""
    ckt = _circulator()
    BA = ckt.orientation_matrix() * ckt.incidence_matrix()
    assert BA == sp.zeros(*BA.shape)


def test_connection_matrix_shape():
    ckt = _circulator()
    M = ckt.connection_matrix()
    assert M.shape == (len(ckt.loops), len(ckt.vertices))


def test_omega_antisymmetric():
    ckt = _circulator()
    Om = ckt.omega()
    assert sp.simplify(Om + Om.T) == sp.zeros(*Om.shape)


def test_circulator_lagrangian_matches_manuscript():
    """The constructed Lagrangian equals the manuscript's, term for term."""
    ckt = _circulator()
    L = sp.expand(ckt.lagrangian())

    phi, q, phidot, qdot = ckt.coordinate_symbols()
    p1, p2, p3 = phi
    Q1, Q2, Q3, Q4 = q
    dp1, dp2, dp3 = phidot
    dQ1, dQ2, dQ3, dQ4 = qdot
    G = sp.Symbol("G")
    EJ = sp.Symbol("E_J")
    C = sp.Symbol("C")

    # the manuscript's Lagrangian (Eq. for the worked example), written out
    L_paper = (
        Q1 * sp.Rational(1, 2) * dp1 + Q2 * sp.Rational(1, 2) * dp1 - Q4 * dp1
        - Q2 * sp.Rational(1, 2) * dp2 - Q3 * sp.Rational(1, 2) * dp2 + Q4 * dp2
        - Q1 * sp.Rational(1, 2) * dp3 + Q3 * sp.Rational(1, 2) * dp3
        + G / 2 * (p1 - p3) * dp2 - G / 2 * (p1 - p3) * dp3
        - sp.Rational(1, 2) / G * (Q1 - Q2) * dQ2
        + sp.Rational(1, 2) / G * (Q1 - Q2) * dQ3
        + EJ * sp.cos(p2 - p1)
        - (Q1 - Q4) ** 2 / (2 * C) - (Q3 - Q4) ** 2 / (2 * C)
    )
    # the package stores the junction term in expanded-trig form
    # (cos(a-b) = cos a cos b + sin a sin b); compare modulo that identity.
    assert sp.simplify(L - L_paper) == 0


def test_circulator_hamiltonian_matches_manuscript():
    """The reduced Hamiltonian equals Eq. (eq:hamiltonian)."""
    ckt = _circulator()
    result = ckt.hamiltonian(ground="v1", open_loops="f4")

    G, C, EJ = sp.Symbol("G"), sp.Symbol("C"), sp.Symbol("E_J")
    phi2, q3 = sp.Symbol("phi_v2"), sp.Symbol("q_f3")
    H_paper = (G * phi2 + q3) ** 2 / (2 * C) + q3 ** 2 / (2 * C) - EJ * sp.cos(phi2)

    assert sp.expand(result.H - H_paper) == 0
    assert result.complete
    # the single constraint is a Noether constraint, the pair is canonical
    assert [kind for kind, _ in result.constraints] == ["noether"]
    pair_coords = {(a, b) for a, b, _ in result.conjugate_pairs}
    pair_coords |= {(b, a) for a, b, _ in result.conjugate_pairs}
    assert (phi2, q3) in pair_coords


def test_pure_null_vector_parallel_capacitors():
    """A pure-capacitor loop gives a *pure null-vector* constraint.

    Two capacitors in parallel across an inductor make the connection matrix
    singular; the pure null vector of Omega imposes the capacitor-loop voltage
    law and combines the capacitances.  In canonical coordinates the result is
    the LC oscillator with ``C_eff = C1 + C2``.
    """
    ckt = Circuit()
    ckt.add_inductor("e1", "v1", "v2", L="L")
    ckt.add_capacitor("e2", "v1", "v2", C="C1")
    ckt.add_capacitor("e3", "v1", "v2", C="C2")
    ckt.add_loop("f1", ["+e1", "-e2"])
    ckt.add_loop("f2", ["+e2", "-e3"])   # pure capacitive loop
    ckt.add_loop("f3", ["-e1", "+e3"])   # outer

    result = ckt.hamiltonian(ground="v1", open_loops="f3")

    # the constraint must be classified as a pure null-vector constraint
    assert [kind for kind, _ in result.constraints] == ["null-vector"]
    assert result.complete

    # in canonical coordinates H is the LC oscillator with C_eff = C1 + C2
    L, C1, C2 = sp.symbols("L C1 C2")
    (a, b, coeff), = result.conjugate_pairs
    charge = b if b in result.H.free_symbols else a
    flux = a if charge is b else b
    Qc = sp.Symbol("Q_c")
    H_canon = sp.simplify(result.H.subs(charge, Qc / coeff))
    H_expected = flux ** 2 / (2 * L) + Qc ** 2 / (2 * (C1 + C2))
    assert sp.simplify(H_canon - H_expected) == 0


def test_lc_oscillator():
    """Bare LC loop reduces to the harmonic oscillator Hamiltonian."""
    ckt = Circuit()
    ckt.add_inductor("e1", "v1", "v2", L="L")
    ckt.add_capacitor("e2", "v2", "v1", C="C")
    ckt.add_loop("f1", ["+e1", "+e2"])

    result = ckt.hamiltonian(ground="v1")
    L, C = sp.Symbol("L"), sp.Symbol("C")
    phi2, q1 = sp.Symbol("phi_v2"), sp.Symbol("q_f1")
    H_expected = phi2 ** 2 / (2 * L) + q1 ** 2 / (2 * C)
    assert sp.expand(result.H - H_expected) == 0


def test_transmon():
    """JJ shunted by a capacitor -> H = Q^2/2C - E_J cos(phi)."""
    ckt = Circuit()
    ckt.add_josephson("e1", "v1", "v2", EJ="E_J")
    ckt.add_capacitor("e2", "v1", "v2", C="C")
    ckt.add_loop("f1", ["+e1", "-e2"])

    result = ckt.hamiltonian(ground="v1")
    C, EJ = sp.Symbol("C"), sp.Symbol("E_J")
    phi, q = sp.Symbol("phi_v2"), sp.Symbol("q_f1")
    H_expected = q ** 2 / (2 * C) - EJ * sp.cos(phi)
    assert result.complete
    assert sp.simplify(result.H - H_expected) == 0


def test_fluxonium():
    """JJ shunted by an inductor and a capacitor ->
    H = Q^2/2C + phi^2/2L - E_J cos(phi)."""
    ckt = Circuit()
    ckt.add_josephson("e1", "v1", "v2", EJ="E_J")
    ckt.add_inductor("e2", "v1", "v2", L="L")
    ckt.add_capacitor("e3", "v1", "v2", C="C")
    ckt.add_loop("f1", ["+e1", "-e2"])
    ckt.add_loop("f2", ["+e2", "-e3"])
    ckt.add_loop("f3", ["-e1", "+e3"])

    result = ckt.hamiltonian(ground="v1", open_loops="f3")
    C, L, EJ = sp.Symbol("C"), sp.Symbol("L"), sp.Symbol("E_J")
    phi, q = sp.Symbol("phi_v2"), sp.Symbol("q_f2")
    H_expected = q ** 2 / (2 * C) + phi ** 2 / (2 * L) - EJ * sp.cos(phi)
    assert result.complete
    assert sp.simplify(result.H - H_expected) == 0


def test_quantum_phase_slip():
    """QPS (charge-space dual of the JJ) shunting an inductor ->
    H = phi^2/2L - E_S cos(q), the charge-space dual of the transmon."""
    ckt = Circuit()
    ckt.add_qps("e1", "v1", "v2", ES="E_S")
    ckt.add_inductor("e2", "v1", "v2", L="L")
    ckt.add_loop("f1", ["+e1", "-e2"])

    result = ckt.hamiltonian(ground="v1")
    L, ES = sp.Symbol("L"), sp.Symbol("E_S")
    phi, q = sp.Symbol("phi_v2"), sp.Symbol("q_f1")
    H_expected = phi ** 2 / (2 * L) - ES * sp.cos(q)
    assert result.complete
    assert sp.simplify(result.H - H_expected) == 0


def test_canonical_form_rescales_pairs():
    """canonical() rescales a non-unit conjugate pair to unit coefficient and
    leaves the physics unchanged (parallel caps -> C1 + C2)."""
    ckt = Circuit()
    ckt.add_inductor("e1", "v1", "v2", L="L")
    ckt.add_capacitor("e2", "v1", "v2", C="C1")
    ckt.add_capacitor("e3", "v1", "v2", C="C2")
    ckt.add_loop("f1", ["+e1", "-e2"])
    ckt.add_loop("f2", ["+e2", "-e3"])
    ckt.add_loop("f3", ["-e1", "+e3"])

    raw = ckt.hamiltonian(ground="v1", open_loops="f3")
    assert not raw.is_canonical
    can = raw.canonical()
    assert can.is_canonical
    L, C1, C2 = sp.symbols("L C1 C2")
    phi, q = sp.Symbol("phi_v2"), sp.Symbol("q_f2")
    H_expected = phi ** 2 / (2 * L) + q ** 2 / (2 * (C1 + C2))
    assert sp.simplify(can.H - H_expected) == 0


def test_validate_rejects_non_cycle_loop():
    """A declared loop that is not a closed cycle is rejected."""
    ckt = Circuit()
    ckt.add_inductor("e1", "v1", "v2", L="L")
    ckt.add_capacitor("e2", "v2", "v3", C="C")
    ckt.add_loop("f1", ["+e1", "+e2"])   # v1->v2->v3, not closed
    with pytest.raises(ValueError):
        ckt.validate()
    with pytest.raises(ValueError):
        ckt.hamiltonian(ground="v1")


def test_validate_rejects_empty_circuit():
    with pytest.raises(ValueError):
        Circuit().validate()


def test_netlist_round_trip():
    """Parsing the circulator netlist reproduces the manuscript Hamiltonian."""
    from fluxcharge import from_netlist
    text = """
    title Circulator
    J    e1  v1 v2  E_J
    C    e2  v2 v3  C
    C    e3  v3 v1  C
    gyrator  e4 v1 v3   e5 v2 v3   G
    loop f1  +e3 +e4
    loop f2  +e1 -e4 +e5
    loop f3  +e2 -e5
    loop f4  -e1 -e2 -e3
    ground v1
    open   f4
    """
    ckt = from_netlist(text)
    assert ckt.ground == "v1"
    assert ckt.open_loops == ["f4"]
    result = ckt.hamiltonian(ground=ckt.ground, open_loops=ckt.open_loops)
    G, C, EJ = sp.symbols("G C E_J")
    phi2, q3 = sp.symbols("phi_v2 q_f3")
    H_pub = (G * phi2 + q3) ** 2 / (2 * C) + q3 ** 2 / (2 * C) - EJ * sp.cos(phi2)
    assert sp.expand(result.H - H_pub) == 0
    assert result.complete


def test_commutation_relations():
    """Transmon: the reduced symplectic form gives [phi, q] = i*hbar, the
    manuscript's canonical convention, and phi is flagged as compact (it sits
    inside the cosine)."""
    ckt = Circuit()
    ckt.add_josephson("e1", "v1", "v2", EJ="E_J")
    ckt.add_capacitor("e2", "v1", "v2", C="C")
    ckt.add_loop("f1", ["+e1", "-e2"])

    result = ckt.hamiltonian(ground="v1")
    comm = result.commutators()
    assert len(comm) == 1
    a, b, val = comm[0]
    hbar = sp.Symbol("hbar", positive=True)
    assert sp.simplify(val - sp.I * hbar) == 0
    assert sp.Symbol("phi_v2") in result.compact_coordinates()


def test_open_loop_choice_does_not_affect_completeness():
    """One face is always redundant (f1+f2+f3+f4 = 0), so the reduction must
    complete to the same 2D phase space whichever loop is opened -- or none."""
    def circ():
        c = Circuit()
        c.add_josephson("e1", "v1", "v2", EJ="E_J")
        c.add_capacitor("e2", "v2", "v3", C="C")
        c.add_capacitor("e3", "v3", "v1", C="C")
        c.add_gyrator(("e4", "v1", "v3"), ("e5", "v2", "v3"), G="G")
        for n, e in [("f1", ["+e3", "+e4"]), ("f2", ["+e1", "-e4", "+e5"]),
                     ("f3", ["+e2", "-e5"]), ("f4", ["-e1", "-e2", "-e3"])]:
            c.add_loop(n, e)
        return c

    for open_loops in ["f1", "f2", "f3", "f4", None]:
        result = circ().hamiltonian(ground="v1", open_loops=open_loops, canonical=True)
        assert result.complete, f"incomplete for open={open_loops}"
        assert len(result.coordinates) == 2
        assert len(result.conjugate_pairs) == 1


def test_lcg_duality():
    """The LCG dual swaps C<->L and JJ<->QPS, inverts gyration ratios to -1/G,
    exchanges vertices<->faces, reduces completely, and is an involution."""
    from fluxcharge import dual
    from fluxcharge.elements import (Capacitor, Inductor, JosephsonJunction,
                                     QuantumPhaseSlip, Gyrator)

    def circ():
        c = Circuit()
        c.add_josephson("e1", "v1", "v2", EJ="E_J")
        c.add_capacitor("e2", "v2", "v3", C="C")
        c.add_capacitor("e3", "v3", "v1", C="C")
        c.add_gyrator(("e4", "v1", "v3"), ("e5", "v2", "v3"), G="G")
        for n, e in [("f1", ["+e3", "+e4"]), ("f2", ["+e1", "-e4", "+e5"]),
                     ("f3", ["+e2", "-e5"]), ("f4", ["-e1", "-e2", "-e3"])]:
            c.add_loop(n, e)
        return c

    c = circ()
    d = dual(c)
    kinds = {type(el).__name__ for el in d._elements}
    assert "QuantumPhaseSlip" in kinds and "Inductor" in kinds  # JJ->QPS, C->L
    assert "Capacitor" not in kinds and "JosephsonJunction" not in kinds
    # dual vertices are the original faces; dual loops are the original vertices
    assert set(d.vertices) == set(c.loops)
    assert set(d.loops) == set(c.vertices)
    # gyration ratio inverted to -1/G
    G = sp.Symbol("G")
    g = next(el for el in d._elements if isinstance(el, Gyrator))
    assert sp.simplify(g.G + 1 / G) == 0
    # the dual reduces to a complete Hamiltonian
    assert dual(c).hamiltonian(strict=False, canonical=True).complete
    # involution: dual of dual restores the original element classes
    dd = dual(d)
    assert {type(el).__name__ for el in dd._elements} == {type(el).__name__ for el in c._elements}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")

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


def _require_numpy():
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        if hasattr(pytest, "skip"):
            pytest.skip("numpy not installed")
        raise SystemExit(0)


def test_numerics_lc_oscillator_spectrum():
    """The LC oscillator's numeric spectrum is omega*(n+1/2), omega=1/sqrt(LC)."""
    _require_numpy()
    import numpy as np
    ckt = Circuit()
    ckt.add_inductor("e1", "v1", "v2", L="L")
    ckt.add_capacitor("e2", "v2", "v1", C="C")
    ckt.add_loop("f1", ["+e1", "+e2"])
    res = ckt.hamiltonian(ground="v1")
    ev = res.eigenenergies({"L": 1.0, "C": 1.0}, n_levels=6, cutoffs={"phi_v2": 60})
    assert np.allclose(ev, (np.arange(6) + 0.5), atol=1e-6)


def test_numerics_mode_types():
    """Mode-type detection: transmon PERIODIC, QPS DUAL_PERIODIC, fluxonium/LC EXTENDED."""
    _require_numpy()
    from fluxcharge.numerics import EXTENDED, PERIODIC, DUAL_PERIODIC

    t = Circuit()
    t.add_josephson("e1", "v1", "v2", EJ="E_J")
    t.add_capacitor("e2", "v1", "v2", C="C")
    t.add_loop("f1", ["+e1", "-e2"])
    assert t.hamiltonian(ground="v1").modes()[0].kind == PERIODIC

    q = Circuit()
    q.add_qps("e1", "v1", "v2", ES="E_S")
    q.add_inductor("e2", "v1", "v2", L="L")
    q.add_loop("f1", ["+e1", "-e2"])
    assert q.hamiltonian(ground="v1").modes()[0].kind == DUAL_PERIODIC

    lc = Circuit()
    lc.add_inductor("e1", "v1", "v2", L="L")
    lc.add_capacitor("e2", "v2", "v1", C="C")
    lc.add_loop("f1", ["+e1", "+e2"])
    assert lc.hamiltonian(ground="v1").modes()[0].kind == EXTENDED


def test_numerics_transmon_qps_duality():
    """A transmon (charge basis) and its LCG dual -- a QPS shunting an inductor
    (flux basis) -- have identical spectra, validating both bases and the
    cosine handling against each other."""
    _require_numpy()
    import numpy as np
    t = Circuit()
    t.add_josephson("e1", "v1", "v2", EJ="E_J")
    t.add_capacitor("e2", "v1", "v2", C="C")
    t.add_loop("f1", ["+e1", "-e2"])
    ev_t = t.hamiltonian(ground="v1").eigenenergies(
        {"E_J": 10.0, "C": 1.0}, n_levels=6, cutoffs={"q_f1": 81})

    q = Circuit()
    q.add_qps("e1", "v1", "v2", ES="E_S")
    q.add_inductor("e2", "v1", "v2", L="L")
    q.add_loop("f1", ["+e1", "-e2"])
    ev_q = q.hamiltonian(ground="v1").eigenenergies(
        {"E_S": 10.0, "L": 1.0}, n_levels=6, cutoffs={"phi_v2": 80})

    assert np.allclose(ev_t, ev_q, atol=1e-9)


def test_numerics_gyrator_quadratic_exact():
    """A gyrator + LC (junction replaced by an inductor) is an exactly solvable
    quadratic Hamiltonian; the numeric spectrum equals omega*(n+1/2) with
    omega = sqrt(a*c - b^2) -- validating the bilinear gyrator cross term."""
    _require_numpy()
    import numpy as np
    c = Circuit()
    c.add_inductor("e1", "v1", "v2", L="L")
    c.add_capacitor("e2", "v2", "v3", C="C")
    c.add_capacitor("e3", "v3", "v1", C="C")
    c.add_gyrator(("e4", "v1", "v3"), ("e5", "v2", "v3"), G="G")
    for n, e in [("f1", ["+e3", "+e4"]), ("f2", ["+e1", "-e4", "+e5"]),
                 ("f3", ["+e2", "-e5"]), ("f4", ["-e1", "-e2", "-e3"])]:
        c.add_loop(n, e)
    res = c.hamiltonian(ground="v1", open_loops="f4", canonical=True)
    C, L, G = 1.0, 1.0, 0.7
    omega = np.sqrt((G ** 2 / C + 1.0 / L) * (2.0 / C) - (G / C) ** 2)
    ev = res.eigenenergies({"C": C, "L": L, "G": G}, n_levels=6,
                           cutoffs={"phi_v2": 120})
    assert np.allclose(ev, omega * (np.arange(6) + 0.5), atol=1e-6)


def test_numerics_circulator_runs_and_converges():
    """The full non-reciprocal circulator diagonalizes to a real, convergent
    spectrum (gyrator cross term + Josephson cosine together)."""
    _require_numpy()
    import numpy as np
    c = Circuit()
    c.add_josephson("e1", "v1", "v2", EJ="E_J")
    c.add_capacitor("e2", "v2", "v3", C="C")
    c.add_capacitor("e3", "v3", "v1", C="C")
    c.add_gyrator(("e4", "v1", "v3"), ("e5", "v2", "v3"), G="G")
    for n, e in [("f1", ["+e3", "+e4"]), ("f2", ["+e1", "-e4", "+e5"]),
                 ("f3", ["+e2", "-e5"]), ("f4", ["-e1", "-e2", "-e3"])]:
        c.add_loop(n, e)
    res = c.hamiltonian(ground="v1", open_loops="f4")
    p = {"E_J": 10.0, "C": 1.0, "G": 0.7}
    ev60 = res.eigenenergies(p, n_levels=3, cutoffs={"phi_v2": 60})
    ev90 = res.eigenenergies(p, n_levels=3, cutoffs={"phi_v2": 90})
    assert np.all(np.isreal(ev90))
    # ground and first excited converge tightly; higher levels a bit slower
    assert np.allclose(ev60[:2], ev90[:2], atol=1e-5)
    assert np.allclose(ev60, ev90, atol=1e-3)


def test_gui_numerical_summary():
    """The headless GUI numerics core classifies and diagonalizes a transmon."""
    _require_numpy()
    from fluxcharge.gui import numerical_summary
    netlist = "J e1 v1 v2 E_J\nC e2 v1 v2 C\nloop f1 +e1 -e2\nground v1"
    s = numerical_summary(netlist, {"E_J": 15.0, "C": 1.0}, n_levels=4)
    assert s["single_mode"]
    assert s["modes"][0][2] == "periodic"
    assert len(s["eigenenergies"]) == 4
    assert len(s["transitions"]) == 3
    # transmon E_01 approaches sqrt(8 E_J E_C) - E_C with E_C = 1/(8C)
    import numpy as np
    EC = 1.0 / 8.0
    assert abs(s["transitions"][0] - (np.sqrt(8 * 15.0 * EC) - EC)) < 0.05


def test_numerics_default_cutoffs_all_mode_kinds():
    """Every mode kind diagonalizes with the default basis size (no cutoffs=),
    including DUAL_PERIODIC (a quantum phase slip)."""
    _require_numpy()
    import numpy as np
    q = Circuit()
    q.add_qps("e1", "v1", "v2", ES="E_S")
    q.add_inductor("e2", "v1", "v2", L="L")
    q.add_loop("f1", ["+e1", "-e2"])
    ev = q.hamiltonian(ground="v1").eigenenergies({"E_S": 10.0, "L": 1.0}, n_levels=4)
    assert len(ev) == 4 and np.all(np.isreal(ev))


def test_gui_summary_from_result_matches():
    """summary_from_result (the cached-reduction path the UI uses) agrees with
    the full numerical_summary."""
    _require_numpy()
    import numpy as np
    from fluxcharge import from_netlist
    from fluxcharge.gui import numerical_summary, summary_from_result
    netlist = "J e1 v1 v2 E_J\nC e2 v1 v2 C\nloop f1 +e1 -e2\nground v1"
    full = numerical_summary(netlist, {"E_J": 12.0, "C": 1.0}, n_levels=5)
    res = from_netlist(netlist).hamiltonian(ground="v1")
    cached = summary_from_result(res, {"E_J": 12.0, "C": 1.0}, n_levels=5)
    assert np.allclose(full["eigenenergies"], cached["eigenenergies"])
    assert full["modes"] == cached["modes"]


def test_numerics_requires_complete_canonical():
    """Diagonalization refuses an incomplete reduction."""
    _require_numpy()
    from fluxcharge.numerics import hamiltonian_matrix
    from fluxcharge.reduction import ReductionResult
    import sympy as sp
    bad = ReductionResult(H=sp.Symbol("phi_v1") ** 2, coordinates=[sp.Symbol("phi_v1")],
                          conjugate_pairs=[], complete=False)
    with pytest.raises(Exception):
        hamiltonian_matrix(bad, {})


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")

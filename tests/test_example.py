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
    # the gyrator's Noether constraint is found (alongside benign redundant-
    # direction eliminations from completing the reduction), and the surviving
    # pair (phi_v2, q_f3) is canonical
    assert "noether" in [kind for kind, _ in result.constraints]
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


def test_canonicalizer_contract_T1():
    """canonical_from_bracket brings any antisymmetric non-degenerate bracket
    matrix to the standard symplectic form J (random sizes up to 2n=8)."""
    _require_numpy()
    import numpy as np
    from fluxcharge.canonicalize import canonical_from_bracket
    rng = np.random.default_rng(1)
    for n in (1, 2, 3, 4):
        Ar = rng.standard_normal((2 * n, 2 * n))
        Pi = Ar - Ar.T
        T = canonical_from_bracket(Pi)
        J = np.block([[np.zeros((n, n)), np.eye(n)],
                      [-np.eye(n), np.zeros((n, n))]])
        assert np.allclose(T @ Pi @ T.T, J, atol=1e-7)


def test_symplectic_eigenvalue_oracle_nonreciprocal_T2():
    """The full symplectic canonicalization reproduces the convention-free
    Williamson frequencies for a NON-reciprocal (gyrator-coupled) 2-mode system
    -- where the symplectic form has nonzero flux-flux/charge-charge blocks --
    while the per-pair shortcut (zeroing those blocks) does not.  Exact oracle
    from the project notes: {0.6431, 2.0094}."""
    _require_numpy()
    import numpy as np
    from fluxcharge.canonicalize import symplectic_eigenvalues
    np.seterr(all="ignore")
    FF = np.array([[0, 0.8], [-0.8, 0]])
    CC = np.array([[0, 0.3], [-0.3, 0]])
    M = np.array([[1, 0.2], [0.1, 1.1]])
    f = np.block([[FF, -M.T], [M, CC]])
    K = np.block([[np.array([[1, 0.1], [0.1, 1.2]]), np.zeros((2, 2))],
                  [np.zeros((2, 2)), np.array([[1.1, 0], [0, 0.9]])]])
    full = np.sort(symplectic_eigenvalues(np.linalg.inv(f), K))
    fpp = np.block([[np.zeros((2, 2)), -M.T], [M, np.zeros((2, 2))]])
    perpair = np.sort(symplectic_eigenvalues(np.linalg.inv(fpp), K))
    assert np.allclose(full, [0.6431, 2.0094], atol=1e-3)
    assert not np.allclose(full, perpair, atol=1e-2)   # per-pair is wrong


def test_gauge_lattice_compactness_T3():
    """Compactness is computed from the circuit's gauge lattice, not guessed:
    transmon -> phi compact, fluxonium -> phi extended, 0-pi -> one compact
    (theta) flux mode."""
    from fluxcharge import library
    from fluxcharge.canonicalize import compact_flux_modes
    assert compact_flux_modes(library.transmon()) == 1
    assert compact_flux_modes(library.fluxonium()) == 0
    assert compact_flux_modes(library.zero_pi()) == 1


def _require_scqubits():
    _require_numpy()
    try:
        import scqubits  # noqa: F401
    except ModuleNotFoundError:
        if hasattr(pytest, "skip"):
            pytest.skip("scqubits not installed")
        raise SystemExit(0)


def test_to_scqubits_export_refuses_nonreciprocal():
    """The scqubits exporter refuses gyrators / quantum phase slips -- scqubits
    has no representation for a non-reciprocal element or a cosine-of-charge."""
    from fluxcharge import library
    from fluxcharge.interop import to_scqubits_yaml
    with pytest.raises(NotImplementedError):
        to_scqubits_yaml(library.circulator(), {"E_J": 10.0, "C": 1.0, "G": 0.5})
    with pytest.raises(NotImplementedError):
        to_scqubits_yaml(library.phase_slip_qubit(charge_bias=False),
                         {"E_S": 10.0, "L": 1.0})


def test_scqubits_cross_check_transmon_clean_oracle():
    """For a charge-network circuit (no linear inductor) scqubits' general
    Circuit class is a clean oracle: the transmon round-trips to ~1e-13.  This
    is the cross-validation harness that will track the multi-mode fix."""
    _require_scqubits()
    from fluxcharge import library
    from fluxcharge.interop import cross_check_spectrum
    r = cross_check_spectrum(library.transmon(), {"E_J": 10.0, "C": 1.0},
                             n_levels=5, ground="v1", cutoffs={"q_f1": 81},
                             scqubits_cutoff=60)
    assert r["max_abs_diff"] < 1e-9


def test_move_across_gyrator_preserves_spectrum():
    """The partial-dual single-element move (Tellegen) is spectrum-preserving: a
    capacitor terminating one gyrator port, moved across, becomes an inductor
    L = C/G^2 on the other port (the emptied gyrator removed by terminated
    deletion), and the circuit's spectrum is unchanged."""
    _require_numpy()
    import numpy as np
    from fluxcharge import Circuit, move_across_gyrator

    def build():
        A = Circuit()
        A.add_capacitor("c0", "a", "g", C="C0")
        A.add_capacitor("cb", "b", "g", C="C")
        A.add_gyrator(("e1", "a", "g"), ("e2", "b", "g"), G="G")
        A.add_loop("f1", ["+c0", "-e1"])
        A.add_loop("f2", ["+cb", "-e2"])
        A.ground = "g"
        return A

    A = build()
    rA = A.hamiltonian(strict=False, canonical=True)
    B = move_across_gyrator(A, "cb")
    # the moved element is now an inductor on the far port; gyrator gone
    kinds = sorted(type(e).__name__ for e in B._elements)
    assert kinds == ["Capacitor", "Inductor"]
    rB = B.hamiltonian(strict=False, canonical=True)
    for p in ({"C0": 1.0, "C": 2.0, "G": 1.5}, {"C0": 3.0, "C": 0.7, "G": 0.8}):
        ea = rA.eigenenergies(p, n_levels=5); ea = ea - ea[0]
        eb = rB.eigenenergies(p, n_levels=5); eb = eb - eb[0]
        assert np.allclose(ea, eb, atol=1e-3)

    # nonlinear move: JJ -> QPS, with the gyration ratio in the cosine argument
    # (winding = G). |G| != 1 warns (cos(q/G) is only a standard slip at |G|=1)
    # but still produces the element; |G| = 1 is the clean Tellegen JJ <-> QPS.
    import warnings as _w
    J = Circuit()
    J.add_capacitor("c0", "a", "g", C="C0")
    J.add_josephson("jb", "b", "g", EJ="E_J")
    J.add_gyrator(("e1", "a", "g"), ("e2", "b", "g"), G="G")
    J.add_loop("f1", ["+c0", "-e1"])
    J.add_loop("f2", ["+jb", "-e2"])
    J.ground = "g"
    with _w.catch_warnings(record=True) as w:
        _w.simplefilter("always")
        DJ = move_across_gyrator(J, "jb")            # symbolic G -> warns, no refuse
    assert any("cos(q/G)" in str(x.message) for x in w)
    qps = [e for e in DJ._elements if type(e).__name__ == "QuantumPhaseSlip"]
    assert len(qps) == 1 and qps[0].winding == sp.Symbol("G")  # cos(q/G)

    # at G = 1 the dual is a standard unit phase slip (winding 1, clean Tellegen)
    J1 = Circuit()
    J1.add_capacitor("c0", "a", "g", C="C0")
    J1.add_josephson("jb", "b", "g", EJ="E_J")
    J1.add_gyrator(("e1", "a", "g"), ("e2", "b", "g"), G=1)
    J1.add_loop("f1", ["+c0", "-e1"]); J1.add_loop("f2", ["+jb", "-e2"]); J1.ground = "g"
    D1 = move_across_gyrator(J1, "jb")
    q1 = [e for e in D1._elements if type(e).__name__ == "QuantumPhaseSlip"]
    assert len(q1) == 1 and q1[0].winding == 1


def test_move_across_gyrator_multi_element_block():
    """The multi-element partial-dual move: a PARALLEL reciprocal block
    terminating a gyrator port maps to the SERIES chain of its duals on the far
    port (duality swaps parallel<->series).  A transmon block JJ || C across a
    gyrator becomes a series QPS - L, well-posed, with the same spectrum."""
    _require_numpy()
    import numpy as np
    from fluxcharge import Circuit, move_across_gyrator

    A = Circuit()
    A.add_capacitor("c0", "a", "g", C="C0")
    A.add_josephson("jb", "b", "g", EJ="E_J")     # transmon block: JJ || C_J
    A.add_capacitor("cj", "b", "g", C="C_J")
    A.add_gyrator(("e1", "a", "g"), ("e2", "b", "g"), G=1)
    A.ground = "g"
    rA = A.hamiltonian(strict=False, canonical=True)
    assert rA.complete

    B = move_across_gyrator(A, ["jb", "cj"])       # JJ||C -> QPS--L series
    kinds = sorted(type(e).__name__ for e in B._elements)
    assert kinds == ["Capacitor", "Inductor", "QuantumPhaseSlip"]   # C0, dual(C_J)=L, dual(JJ)=QPS
    # the duals are in series (share an interior node), not in parallel
    qps = next(e for e in B._elements if type(e).__name__ == "QuantumPhaseSlip")
    ind = next(e for e in B._elements if type(e).__name__ == "Inductor")
    assert set((qps._edge.tail, qps._edge.head)) & set((ind._edge.tail, ind._edge.head))
    rB = B.hamiltonian(strict=False, canonical=True)
    assert rB.complete

    p = {"C0": 1.0, "E_J": 6.0, "C_J": 1.5}
    cutA = {str(b): 41 for _a, b, _c in rA.conjugate_pairs}
    cutB = {str(b): 41 for _a, b, _c in rB.conjugate_pairs}
    ea = rA.eigenenergies(p, n_levels=5, cutoffs=cutA); ea = ea - ea[0]
    eb = rB.eigenenergies(p, n_levels=5, cutoffs=cutB); eb = eb - eb[0]
    assert np.allclose(ea, eb, atol=1e-3)

    # a partial block (leaving an element behind on the port) is refused
    import pytest as _pytest
    with _pytest.raises(NotImplementedError):
        move_across_gyrator(A, "jb")


def test_move_across_gyrator_preserves_well_posedness_JJ_to_QPS():
    """Partial duality is a point transformation, so it preserves well-posedness
    (manuscript Sec. "Partial Dual Transformations").  A WELL-POSED junction via
    a gyrator -- an inductor across the partner port giving the JJ an effective
    capacitance -- maps to a well-posed phase-slip circuit with the SAME
    spectrum (the clean JJ <-> QPS Tellegen result at G=1)."""
    _require_numpy()
    import numpy as np
    from fluxcharge import Circuit, move_across_gyrator

    A = Circuit()
    A.add_inductor("l0", "a", "g", L="L0")          # gives the JJ effective C
    A.add_josephson("jb", "b", "g", EJ="E_J")
    A.add_gyrator(("e1", "a", "g"), ("e2", "b", "g"), G=1)
    A.add_loop("f1", ["+l0", "-e1"])
    A.add_loop("f2", ["+jb", "-e2"])
    A.ground = "g"
    rA = A.hamiltonian(strict=False, canonical=True)
    assert rA.complete

    B = move_across_gyrator(A, "jb")               # JJ -> QPS, gyrator deleted
    assert sorted(type(e).__name__ for e in B._elements) == ["Inductor", "QuantumPhaseSlip"]
    rB = B.hamiltonian(strict=False, canonical=True)
    assert rB.complete                              # still well-posed

    p = {"L0": 1.0, "E_J": 8.0}
    cutA = {str(b): 41 for _a, b, _c in rA.conjugate_pairs}
    cutB = {str(b): 41 for _a, b, _c in rB.conjugate_pairs}
    ea = rA.eigenenergies(p, n_levels=5, cutoffs=cutA); ea = ea - ea[0]
    eb = rB.eigenenergies(p, n_levels=5, cutoffs=cutB); eb = eb - eb[0]
    assert np.allclose(ea, eb, atol=1e-3)


def test_move_across_gyrator_carries_offset_charge():
    """External-bias carry-over across the partial-dual move.  An offset (gate)
    charge is a node-local property: when its node survives the move with
    capacitive edges it carries over directly, and the spectrum (a point
    transformation) is preserved.  A bias the move would have to *dualize through*
    the gyrator -- an offset charge on the emptied island, or any loop-keyed flux
    bias against the re-inferred output -- is refused (honest, not silently
    dropped) rather than mis-placed."""
    _require_numpy()
    import numpy as np
    import pytest as _pytest
    from fluxcharge import Circuit, move_across_gyrator

    def build():
        A = Circuit()
        A.add_capacitor("c0", "a", "g", C="C0")
        A.add_capacitor("cb", "b", "g", C="C")
        A.add_gyrator(("e1", "a", "g"), ("e2", "b", "g"), G="G")
        A.add_loop("f1", ["+c0", "-e1"]); A.add_loop("f2", ["+cb", "-e2"])
        A.ground = "g"
        return A

    # offset charge on the surviving far node 'a' carries; spectrum preserved
    A = build(); A.set_offset_charge("a", "n_g")
    B = move_across_gyrator(A, "cb")
    assert dict(B._offset_charge) == {"a": sp.Symbol("n_g")}
    rA = A.hamiltonian(strict=False, canonical=True)
    rB = B.hamiltonian(strict=False, canonical=True)
    for ng in (0.0, 0.4):
        p = {"C0": 1.0, "C": 2.0, "G": 1.0, "n_g": ng}
        ea = rA.eigenenergies(p, n_levels=4, cutoffs={str(b): 80 for _a, b, _c in rA.conjugate_pairs})
        eb = rB.eigenenergies(p, n_levels=4, cutoffs={str(b): 80 for _a, b, _c in rB.conjugate_pairs})
        assert np.allclose(ea - ea[0], eb - eb[0], atol=2e-2)

    # offset charge on the moved island 'b' -> refused (its dual is a flux)
    A = build(); A.set_offset_charge("b")
    with _pytest.raises(NotImplementedError):
        move_across_gyrator(A, "cb")

    # any flux bias -> refused (output re-infers loops; no stable target)
    A = build(); A.set_flux_bias("f1")
    with _pytest.raises(NotImplementedError):
        move_across_gyrator(A, "cb")


def test_interactive_spectrum_slider_updates():
    """The self-contained slider explorer builds for any circuit and recomputes
    the spectrum when a parameter slider moves -- including the gyrator circuits
    scqubits cannot represent.  Headless backend: no window, but the figure is
    built and one update has run."""
    _require_numpy()
    import matplotlib
    matplotlib.use("Agg")
    from fluxcharge import library
    from fluxcharge.interactive import spectrum_slider, parameter_symbols

    r = library.transmon().hamiltonian(ground="v1")
    assert sorted(map(str, parameter_symbols(r))) == ["C", "E_J"]
    fig, sliders = spectrum_slider(r, {"E_J": (1, 30), "C": (0.2, 2.0)},
                                   n_levels=4, show=False)
    assert set(sliders) == {"E_J", "C"}
    y_before = [ln.get_ydata()[0] for ln in fig.axes[0].lines]
    sliders["E_J"].set_val(25.0)
    y_after = [ln.get_ydata()[0] for ln in fig.axes[0].lines]
    assert y_before != y_after            # the spectrum moved with the slider
    assert y_after[0] == 0.0              # ground state is the reference

    # a gyrator circuit (no scqubits equivalent) still gets a working explorer
    rc = library.circulator().hamiltonian(ground="v1", open_loops="f4", canonical=True)
    _, slc = spectrum_slider(rc, {"E_J": (1, 20), "G": (0.2, 0.9)},
                             n_levels=3, cutoffs={"phi_v2": 60}, show=False)
    assert "G" in slc


def test_interactive_bias_sliders_and_vs_param():
    """Bias-aware ranges + the evals/transitions-vs-parameter view.  An external
    flux defaults to a 0..2*pi sweep and is auto-chosen as the x-axis; the
    fluxonium spectrum then modulates with it.  An offset charge defaults to
    0..1, and the transitions view plots f_{i,i+1}."""
    _require_numpy()
    import math
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    from fluxcharge import library
    from fluxcharge.interactive import (spectrum_vs_param, _default_ranges,
                                        is_flux_bias, is_charge_bias)

    # bias-aware default spans: flux 0..2pi (sweet spot pi), charge 0..1
    fx = library.fluxonium().hamiltonian(ground="v1", open_loops="f3")
    dr = _default_ranges(fx)
    assert is_flux_bias("phi_ext_f1") and not is_charge_bias("phi_ext_f1")
    assert dr["phi_ext_f1"] == (0.0, 2 * math.pi, math.pi)

    # auto-sweep picks the flux bias; the spectrum modulates with it
    fig, sl = spectrum_vs_param(fx, ranges={"E_J": (1, 8, 5)},
                                n_levels=3, cutoffs={"phi_v2": 60},
                                npoints=15, show=False)
    assert "phi_ext_f1" in sl                       # sweep slider present
    y1 = fig.axes[0].lines[1].get_ydata()           # |1> curve over the flux sweep
    assert np.nanmax(y1) - np.nanmin(y1) > 0.05     # it actually modulates

    # transitions view: one fewer curve than levels (consecutive gaps)
    cpb = library.cooper_pair_box().hamiltonian(ground="v1")
    figc, _ = spectrum_vs_param(cpb, sweep="n_g_v2", quantity="transitions",
                                ranges={"E_J": (1, 15, 1), "C": (0.3, 2, 1)},
                                n_levels=3, cutoffs={"q_f1": 61}, npoints=15, show=False)
    # 2 transition curves (f01, f12) for 3 levels; charge dispersion visible at E_J=1
    curves = [ln for ln in figc.axes[0].lines if ln.get_label().startswith("$f")]
    assert len(curves) == 2
    assert np.nanmax(curves[0].get_ydata()) - np.nanmin(curves[0].get_ydata()) > 1e-3


def test_interactive_transition_strength_weighting():
    """weight_by colours each curve by a transition matrix element |<i|op|j>|.
    The transmon charge matrix element grows with E_J/E_C, so the f01 strength
    is non-trivial and varies over an E_J sweep.  Labels render the bias physics
    names cleanly."""
    _require_numpy()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.collections import LineCollection
    from fluxcharge import library
    from fluxcharge.interactive import spectrum_vs_param, _pretty_label, _default_drive

    assert _pretty_label("E_J") == "$E_{J}$"
    assert "Phi" in _pretty_label("phi_ext_f1") and "ext" in _pretty_label("phi_ext_f1")
    assert "n_{g}" in _pretty_label("n_g_v2")

    tr = library.transmon().hamiltonian(ground="v1")
    assert _default_drive(tr) == "q_f1"          # natural charge drive
    fig, _ = spectrum_vs_param(tr, sweep="E_J", quantity="transitions", weight_by=True,
                               ranges={"E_J": (1, 30, 12), "C": (0.3, 2, 1)},
                               n_levels=3, cutoffs={"q_f1": 61}, npoints=15, show=False)
    lcs = [c for c in fig.axes[0].collections if isinstance(c, LineCollection)]
    assert len(lcs) == 2                          # f01, f12 as coloured collections
    f01 = lcs[0].get_array()
    assert np.nanmax(f01) - np.nanmin(f01) > 1e-2 # strength varies over the sweep


def test_interactive_auto_cutoffs_basis_aware():
    """auto_cutoffs picks the basis each mode is diagonalized in: a charge basis
    for a PERIODIC (transmon-like) mode -- keyed by the charge, odd size -- and a
    flux/Fock basis for an EXTENDED (fluxonium-like) mode, keyed by the flux."""
    _require_numpy()
    from fluxcharge import library
    from fluxcharge.interactive import auto_cutoffs

    tr = library.transmon(); tr.set_offset_charge("v2")
    ct = auto_cutoffs(tr.hamiltonian(ground="v1", canonical=True))
    assert set(ct) == {"q_f1"} and ct["q_f1"] % 2 == 1      # charge basis, odd

    fx = auto_cutoffs(library.fluxonium().hamiltonian(
        ground="v1", open_loops="f3", canonical=True))
    assert set(fx) == {"phi_v2"}                            # flux/Fock basis


def test_gyrator_terminated_capacitor_is_lc_mode():
    """A gyrator terminated by a capacitor presents an inductance L = C/G^2
    (Tellegen): with a shunt C0 the circuit is a single LC oscillator with
    omega = G/sqrt(C0 C).  Regression for the cyclic-drop bug, where the node
    fluxes (which sit undifferentiated in the gyrator flux-sector but are absent
    from the purely-capacitive energy) were wrongly dropped as cyclic, deleting
    the mode and leaving zero coordinates."""
    _require_numpy()
    import numpy as np
    from fluxcharge import Circuit
    A = Circuit()
    A.add_capacitor("c0", "a", "g", C="C0")
    A.add_capacitor("cb", "b", "g", C="C")
    A.add_gyrator(("e1", "a", "g"), ("e2", "b", "g"), G="G")
    A.add_loop("f1", ["+c0", "-e1"])
    A.add_loop("f2", ["+cb", "-e2"])
    A.ground = "g"
    r = A.hamiltonian(strict=False, canonical=True)
    assert r.complete
    assert len(r.modes()) == 1                       # one LC mode, not zero
    # no eliminated coordinate leaks into H (chained-Noether resolution)
    assert not (r.H.free_symbols & {sp.Symbol("q_f1"), sp.Symbol("q_f2")})
    p = {"C0": 1.0, "C": 2.0, "G": 1.5}
    ev = r.eigenenergies(p, n_levels=4)
    ev = ev - ev[0]
    omega = p["G"] / np.sqrt(p["C0"] * p["C"])
    assert np.allclose(ev, np.arange(4) * omega, atol=1e-3)


def test_gyrator_nonlinear_circular_constraint_guards():
    """A gyrator coupling a *nonlinear* element to an element of the same kind
    (here a phase slip across the gyrator from an inductor) is ill-posed and its
    constraint analysis is a transcendental self-consistency ``x = f(sin x)``
    with no closed form.  The reducer must raise ``ReductionError`` promptly
    rather than hang building an infinitely-nested ``sin(sin(...))`` (the old
    behaviour) -- and the partial-dual move across it stays a valid structural
    transform whose (also ill-posed) output guards the same way.  Regression for
    the QPS+gyrator reduction hang."""
    import signal
    from fluxcharge import Circuit, move_across_gyrator
    from fluxcharge.reduction import ReductionError

    A = Circuit()
    A.add_inductor("l0", "a", "g", L="L0")
    A.add_qps("qb", "b", "g", ES="E_S")
    A.add_gyrator(("e1", "a", "g"), ("e2", "b", "g"), G=1)
    A.ground = "g"

    def _bail(signum, frame):
        raise AssertionError("reduction hung instead of raising ReductionError")
    old = signal.signal(signal.SIGALRM, _bail)
    signal.alarm(30)
    try:
        with pytest.raises(ReductionError):
            A.hamiltonian(strict=False)
        # the move itself is purely structural and still succeeds: QPS -> JJ on
        # the far port, gyrator removed (the output is the ill-posed JJ||L)
        B = move_across_gyrator(A, "qb")
        assert sorted(type(e).__name__ for e in B._elements) == \
            ["Inductor", "JosephsonJunction"]
        with pytest.raises(ReductionError):
            B.hamiltonian(strict=False)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def test_gui_mode_type_options():
    """The diagonalize dialog's data: per conjugate pair a (flux, charge,
    default_kind, warning).  Defaults are the auto classification; a warning
    fires only for the subtle cases (a coordinate inside a nonlinear cosine ->
    coordinate-dependent bracket)."""
    _require_numpy()
    from fluxcharge import library, Circuit
    from fluxcharge.gui import mode_type_options

    tr = mode_type_options(library.transmon().hamiltonian(ground="v1", canonical=True))
    assert len(tr) == 1 and tr[0][2] == "periodic" and tr[0][3] is None  # no warning

    fx = mode_type_options(library.fluxonium().hamiltonian(
        ground="v1", open_loops="f3", canonical=True))
    assert fx[0][2] == "extended" and fx[0][3] is None

    # multi-mode non-reciprocal: coordinate-dependent bracket -> warned
    B = Circuit()
    B.add_capacitor("c0", "a", "g", C="C0")
    B.add_inductor("lq", "a", "g", L="Lq")
    B.add_josephson("jb", "b", "g", EJ="E_J")
    B.add_gyrator(("e1", "a", "g"), ("e2", "b", "g"), G=1)
    B.ground = "g"
    rows = mode_type_options(B.hamiltonian(strict=False, canonical=True))
    assert any(w is not None for *_x, w in rows)


def test_gui_default_params_prefill():
    """The diagonalization box pre-fills a default value for every parameter the
    circuit needs (biases -> 0), keeps values the user already typed, and can
    emit physical-unit defaults."""
    from fluxcharge import library
    from fluxcharge.gui import default_params, param_entry_text

    tr = library.transmon().hamiltonian(ground="v1", canonical=True)
    assert default_params(tr) == {"C": 1, "E_J": 15}

    fx = library.fluxonium().hamiltonian(ground="v1", open_loops="f3", canonical=True)
    dfx = default_params(fx)
    assert dfx["E_J"] == 15 and dfx["L"] == 1 and dfx["phi_ext_f1"] == 0  # bias -> 0

    cir = library.circulator().hamiltonian(ground="v1", open_loops="f4", canonical=True)
    assert default_params(cir)["G"] == 0.5

    # preserve a value the user already typed; default the rest
    txt = param_entry_text(fx, existing="E_J=7")
    assert "E_J=7" in txt and "L=1" in txt
    # physical-units defaults carry suffixes
    assert "70fF" in param_entry_text(fx, physical=True)


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


def test_dual_completes_outer_face_transmon():
    """A circuit that declares only its inner face(s) -- e.g. the transmon, with
    one loop -- still dualizes: the missing outer face is completed
    automatically. The dual reduces to a Hamiltonian and (being a unitary map)
    shares the original's spectrum; dualizing twice returns to it."""
    from fluxcharge import dual

    t = Circuit()
    t.add_josephson("e1", "v1", "v2", EJ="E_J")
    t.add_capacitor("e2", "v1", "v2", C="C")
    t.add_loop("f1", ["+e1", "-e2"])          # only the inner face declared

    d = dual(t)
    assert set(d.vertices) == {"f1", "outer"}   # outer face synthesized
    # JJ -> QPS, C -> L
    kinds = {type(el).__name__ for el in d._elements}
    assert "QuantumPhaseSlip" in kinds and "Inductor" in kinds
    dr = d.hamiltonian(strict=False, canonical=True)
    assert dr.complete

    _require_numpy()
    import numpy as np
    ev_t = t.hamiltonian(ground="v1").eigenenergies(
        {"E_J": 10.0, "C": 1.0}, n_levels=6, cutoffs={"q_f1": 81})
    ev_d = np.sort(dr.eigenenergies({"E_J": 10.0, "C": 1.0}, n_levels=6))
    assert np.allclose(ev_t, ev_d, atol=1e-6)   # duality preserves the spectrum
    # involution
    assert {type(el).__name__ for el in dual(d)._elements} == \
        {type(el).__name__ for el in t._elements}


def test_dual_of_every_example_netlist():
    """Every shipped example circuit dualizes and the dual reduces completely."""
    from fluxcharge import from_netlist, dual
    import os
    exdir = os.path.join(os.path.dirname(__file__), os.pardir, "examples")
    for name in ["transmon.txt", "fluxonium.txt", "qps_inductor.txt",
                 "lc_oscillator.txt", "circulator.txt"]:
        path = os.path.join(exdir, name)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            c = from_netlist(fh.read())
        d = dual(c)
        assert d.hamiltonian(strict=False, canonical=True).complete, name


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


def test_latex_wrapping_helpers():
    """A long Hamiltonian splits into top-level terms (not inside fractions) and
    wraps onto multiple lines so it fits the GUI panel."""
    from fluxcharge.gui import _split_latex_terms, wrap_latex_sum
    terms = _split_latex_terms(r"a + b - \frac{c+d}{e} + f")
    assert len(terms) == 4                       # the c+d inside \frac is not split
    assert any("c+d" in t for t in terms)
    long = " + ".join(f"x_{i}" for i in range(30))
    lines = wrap_latex_sum(long, r"\hat{H} =", max_px=60, fontsize=12, dpi=100)
    assert len(lines) > 1 and lines[0].startswith(r"\hat{H} =")


def test_gui_energy_units_form():
    """The familiar-units rewrite turns the transmon into 4 E_C n^2 - E_J cos(phi)
    and the fluxonium into 4 E_C n^2 + E_L phi^2/2 - E_J cos(phi), relabelling the
    charge q -> n in both H and the commutators."""
    from fluxcharge.gui import energy_units_form

    def plain(expr):  # drop symbol assumptions so name-equal symbols cancel
        return expr.xreplace({s: sp.Symbol(s.name) for s in expr.free_symbols})

    EC, EL, EJ = sp.symbols("E_C E_L E_J")
    n, phi = sp.Symbol("n_f1"), sp.Symbol("phi_v2")
    hbar = sp.Symbol("hbar")

    # transmon
    t = Circuit()
    t.add_josephson("e1", "v1", "v2", EJ="E_J")
    t.add_capacitor("e2", "v1", "v2", C="C")
    t.add_loop("f1", ["+e1", "-e2"])
    r = t.hamiltonian(ground="v1")
    H_e, comm_e, defs, cmap = energy_units_form(
        r.H, r.commutators(), {sp.Symbol("C")}, set())
    assert sp.expand(plain(H_e) - (4 * EC * n ** 2 - EJ * sp.cos(phi))) == 0
    # charge relabelled in the commutator, value preserved
    a, b, val = comm_e[0]
    assert b == n and sp.simplify(plain(val) - sp.I * hbar) == 0
    assert "E_C" in [d.lhs.name for d in defs]

    # fluxonium
    fx = Circuit()
    fx.add_josephson("e1", "v1", "v2", EJ="E_J")
    fx.add_inductor("e2", "v1", "v2", L="L")
    fx.add_capacitor("e3", "v1", "v2", C="C")
    fx.add_loop("f1", ["+e1", "-e2"]); fx.add_loop("f2", ["+e2", "-e3"])
    fx.add_loop("f3", ["-e1", "+e3"])
    rf = fx.hamiltonian(ground="v1", open_loops="f3")
    Hf, _, _, _ = energy_units_form(rf.H, rf.commutators(),
                                    {sp.Symbol("C")}, {sp.Symbol("L")})
    nf = sp.Symbol("n_f2")
    assert sp.expand(plain(Hf) - (4 * EC * nf ** 2 + EL * phi ** 2 / 2
                                  - EJ * sp.cos(phi))) == 0


def _charge_cutoffs(result, n=81):
    return {str(b): n for _a, b, _c in result.conjugate_pairs}


def test_scqubits_yaml_import():
    """Import scqubits' branch YAML: a JJ + capacitor transmon and a fluxonium,
    checking the resulting symbolic Hamiltonian and that node 0 is ground."""
    from fluxcharge import from_scqubits_yaml
    EC, EJ, EL = sp.symbols("EC EJ EL")

    # transmon: JJ (no junction cap) shunted by a capacitor, node 0 = ground
    ckt, params = from_scqubits_yaml("branches:\n- [JJ, 1, 0, EJ]\n- [C, 1, 0, EC]\n")
    assert ckt.ground == "0"
    res = ckt.hamiltonian(ground="0", canonical=True)
    (a, b, _), = res.conjugate_pairs
    q = b if str(b).startswith("q_") else a
    phi = a if q is b else b
    # C -> 1/(8 EC) so q^2/2C = 4 EC q^2; matches 4 E_C n^2 - E_J cos(phi)
    assert sp.simplify(res.H - (4 * EC * q ** 2 - EJ * sp.cos(phi))) == 0

    # a JJ branch with a junction EC adds a parallel capacitor (extra edge)
    ckt2, _ = from_scqubits_yaml("branches:\n- [JJ, 1, 0, EJ, EC]\n")
    assert len(ckt2.edges) == 2          # junction + its capacitor

    # fluxonium imports as JJ + L + C with the right parameters present
    ckt3, p3 = from_scqubits_yaml(
        "branches:\n- [JJ,1,2,EJ,1e15]\n- [L,1,2,EL]\n- [C,1,2,EC]\n")
    r3 = ckt3.hamiltonian(ground="1", strict=False, canonical=True)
    assert r3.complete
    syms = {str(s) for s in r3.H.free_symbols}
    assert {"EJ", "EL", "EC"} <= syms

    # unsupported branch type is rejected with a clear error
    with pytest.raises(NotImplementedError):
        from_scqubits_yaml("branches:\n- [ML, 1, 2, 0.1]\n")

    # unit suffixes: a capacitance (fF) -> charging energy, inductance (nH) ->
    # inductive energy, explicit GHz -> energy as-is
    from fluxcharge import charging_energy_GHz, inductive_energy_GHz
    ck, pr = from_scqubits_yaml(
        "branches:\n- [JJ, 1, 0, EJ = 15 GHz, EC = 90 fF]\n- [L, 1, 0, EL = 5 nH]\n")
    assert abs(pr["EJ"] - 15.0) < 1e-9
    assert abs(pr["EC"] - charging_energy_GHz(90.0)) < 1e-6     # 90 fF -> E_C
    assert abs(pr["EL"] - inductive_energy_GHz(5.0)) < 1e-6     # 5 nH  -> E_L


def test_infer_loops_reproduces_hand_declared_spectra():
    """Auto-inferred loops give the same spectrum as hand-declared faces, for
    the transmon, fluxonium and the gyrator circulator."""
    _require_numpy()
    import numpy as np

    def transmon(declare):
        c = Circuit()
        c.add_josephson("e1", "v1", "v2", EJ="E_J")
        c.add_capacitor("e2", "v1", "v2", C="C")
        if declare:
            c.add_loop("f1", ["+e1", "-e2"])
        return c, {"E_J": 12.0, "C": 1.0}

    def fluxonium(declare):
        c = Circuit()
        c.add_josephson("e1", "v1", "v2", EJ="E_J")
        c.add_inductor("e2", "v1", "v2", L="L")
        c.add_capacitor("e3", "v1", "v2", C="C")
        if declare:
            c.add_loop("f1", ["+e1", "-e2"]); c.add_loop("f2", ["+e2", "-e3"])
            c.add_loop("f3", ["-e1", "+e3"])
        return c, {"E_J": 8.0, "L": 1.0, "C": 1.0}

    def circulator(declare):
        c = Circuit()
        c.add_josephson("e1", "v1", "v2", EJ="E_J")
        c.add_capacitor("e2", "v2", "v3", C="C")
        c.add_capacitor("e3", "v3", "v1", C="C")
        c.add_gyrator(("e4", "v1", "v3"), ("e5", "v2", "v3"), G="G")
        if declare:
            for n, e in [("f1", ["+e3", "+e4"]), ("f2", ["+e1", "-e4", "+e5"]),
                         ("f3", ["+e2", "-e5"]), ("f4", ["-e1", "-e2", "-e3"])]:
                c.add_loop(n, e)
        return c, {"E_J": 10.0, "C": 1.0, "G": 0.5}

    for build in (transmon, fluxonium, circulator):
        ch, params = build(True)
        ca, _ = build(False)
        rh = ch.hamiltonian(ground="v1", strict=False, canonical=True)
        ra = ca.hamiltonian(ground="v1", strict=False, canonical=True)
        assert ca._planar is True
        ev_h = rh.eigenenergies(params, n_levels=5, cutoffs=_charge_cutoffs(rh, 60))
        ev_a = ra.eigenenergies(params, n_levels=5, cutoffs=_charge_cutoffs(ra, 60))
        assert np.allclose(ev_h, ev_a, atol=1e-6), build.__name__


def test_infer_loops_nonplanar_fallback():
    """A non-planar circuit (K5 of inductors) yields a cycle basis of E-V+1
    loops that satisfy Kirchhoff (B*A=0), with the non-planar flag set."""
    import sympy as sp
    import warnings
    c = Circuit()
    nodes = ["v1", "v2", "v3", "v4", "v5"]
    k = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            k += 1
            c.add_inductor(f"e{k}", nodes[i], nodes[j], L=f"L{k}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c.infer_loops()
    assert c._planar is False
    assert len(c.loops) == len(c.edges) - len(c.vertices) + 1   # E - V + 1
    BA = c.orientation_matrix() * c.incidence_matrix()
    assert BA == sp.zeros(*BA.shape)                            # valid cycles


def test_hamiltonian_without_declared_loops():
    """hamiltonian() works with no loops declared (auto-inference)."""
    ckt = Circuit()
    ckt.add_josephson("e1", "v1", "v2", EJ="E_J")
    ckt.add_capacitor("e2", "v1", "v2", C="C")
    res = ckt.hamiltonian(ground="v1", canonical=True)   # no add_loop
    assert res.complete
    C, EJ = sp.Symbol("C"), sp.Symbol("E_J")
    phi = sp.Symbol("phi_v2")
    (a, b, _), = res.conjugate_pairs
    q = b if str(b).startswith("q_") else a
    assert sp.simplify(res.H - (q ** 2 / (2 * C) - EJ * sp.cos(phi))) == 0


def test_library_circuits_reduce_and_diagonalize():
    """Every library circuit reduces to a complete Hamiltonian and (where it has
    enough parameters set) diagonalizes to a real spectrum."""
    _require_numpy()
    import numpy as np
    from fluxcharge import library
    defaults = {"E_J": "12GHz", "C": "70fF", "C_J": "30fF", "L": "150nH",
                "E_S": "12GHz", "G": 0.5}
    for name, ctor in library.CIRCUITS.items():
        ckt = ctor()
        res = ckt.hamiltonian(ground=getattr(ckt, "ground", None),
                              open_loops=getattr(ckt, "open_loops", None) or None,
                              canonical=True)
        assert res.complete, name
        needed = {str(s) for s in res.H.free_symbols} - {str(c) for c in res.coordinates}
        phys = {k: v for k, v in defaults.items() if k in needed}
        params = ckt.natural_params(phys)
        for s in needed:
            params.setdefault(s, 0.0)        # bias symbols (n_g_*, phi_ext_*) -> 0
        # small explicit cutoffs so multi-mode circuits (zero-pi) stay fast; we
        # only check the run succeeds and the spectrum is real, not convergence
        cut = {str(b): 6 for _a, b, _c in res.conjugate_pairs}
        from fluxcharge.canonicalize import CompactLatticeError
        try:
            ev = res.eigenenergies(params, n_levels=3, cutoffs=cut)
            assert np.all(np.isreal(ev)), name
        except CompactLatticeError:
            pass  # multi-mode compact frame not auto-quantizable (0-pi) -- guarded


def test_zero_pi_dense_bracket_is_guarded():
    """The 0-pi qubit's symbolic reduction is correct (three modes, two junction
    cosines), but its reduced bracket is **dense** -- the flux<->charge block is
    not block-diagonal in the conjugate pairs (cross-brackets ~0.67) -- and it is
    nonlinear, so the per-pair operator basis would silently drop those
    cross-brackets and give a wrong spectrum.  The numeric layer must refuse
    (CompactLatticeError) rather than return an unjustified number; the symbolic
    Hamiltonian is still the correct deliverable."""
    _require_numpy()
    from fluxcharge import library
    from fluxcharge.numerics import bracket_is_block_diagonal
    from fluxcharge.canonicalize import CompactLatticeError
    res = library.zero_pi().hamiltonian(ground="v1", strict=False, canonical=True)
    assert res.complete
    assert sorted(m.kind for m in res.modes()) == ["extended", "extended", "periodic"]
    assert sum(1 for _ in res.H.atoms(sp.cos)) == 2          # two junctions
    p = {"E_J": 1.0, "C_J": 1.0, "L": 1.0, "C": 1.0}
    assert not bracket_is_block_diagonal(res, p)             # dense flux-charge block
    with pytest.raises(CompactLatticeError):
        res.eigenenergies(p, n_levels=5,
                          cutoffs={str(b): 8 for _a, b, _c in res.conjugate_pairs})


def test_commutators_report_full_bracket_matrix():
    """commutators() reports the full f^{-1}, not a per-pair shortcut: a clean
    single +/- i*hbar for a block-diagonal (single-mode) circuit, and the dense
    cross-brackets for a multi-mode one.  For 0-pi a flux brackets several
    charges, and its naive per-pair partner (phi_v3, q_f3) is actually decoupled
    -- exactly the information that shows the per-pair basis cannot be used."""
    _require_numpy()
    import sympy as sp
    from fluxcharge import library
    hbar = sp.Symbol("hbar", positive=True)

    tr = library.transmon().hamiltonian(ground="v1", canonical=True)
    comm = tr.commutators(hbar)
    assert len(comm) == 1 and sp.Abs(comm[0][2]) == hbar      # [phi, q] = +/- i*hbar

    zp = library.zero_pi().hamiltonian(ground="v1", strict=False, canonical=True)
    pairs = {(str(a), str(b)) for a, b, _ in zp.commutators(hbar)}
    assert ("phi_v3", "q_f1") in pairs and ("phi_v3", "q_f5") in pairs  # dense cross-brackets
    assert ("phi_v3", "q_f3") not in pairs   # naive per-pair partner is decoupled


def test_quadratic_dense_circuit_uses_williamson():
    """A purely *linear* multi-mode circuit with a dense bracket (here a gyrator
    coupling two LC oscillators) is solved exactly from its symplectic normal-mode
    (Williamson) frequencies -- the per-pair basis cannot, but the quadratic
    spectrum is convention-free.  The single-mode LC check pins the frequency to
    1/sqrt(LC)."""
    _require_numpy()
    import numpy as np
    import sympy as sp
    from fluxcharge import library, Circuit
    from fluxcharge.numerics import bracket_is_block_diagonal, _is_quadratic

    # single-mode oracle: an LC oscillator's ladder spacing is 1/sqrt(LC)
    lc = library.lc_resonator().hamiltonian(ground="v1", canonical=True)
    ev = lc.eigenenergies({"L": 2.0, "C": 3.0}, n_levels=4)
    assert np.allclose(np.diff(ev), 1.0 / np.sqrt(6.0), atol=1e-6)

    # dense quadratic multi-mode: gyrator-coupled oscillators -> Williamson path
    c = Circuit()
    c.add_capacitor("ca", "a", "g", C="C"); c.add_inductor("la", "a", "g", L="L")
    c.add_capacitor("cb", "b", "g", C="C"); c.add_inductor("lb", "b", "g", L="L")
    c.add_gyrator(("e1", "a", "g"), ("e2", "b", "g"), G="G"); c.ground = "g"
    rg = c.hamiltonian(strict=False, canonical=True)
    p = {"C": 1.0, "L": 1.0, "G": 0.5}
    assert not bracket_is_block_diagonal(rg, p) and _is_quadratic(sp.expand(rg.H.subs(p)))
    ev = rg.eigenenergies(p, n_levels=5)
    ev = ev - ev[0]
    assert np.all(np.isreal(ev)) and ev[0] == 0 and np.all(np.diff(ev) >= -1e-9)


def test_tidy_hamiltonian_collects_and_preserves_spectrum():
    """tidy_hamiltonian folds each monomial into one factored coefficient (so the
    0-pi H is short) without changing the Hamiltonian it represents, and folds
    Josephson powers back to an integer-argument cosine."""
    _require_numpy()
    import numpy as np
    import sympy as sp
    from fluxcharge import library, Circuit, dual
    from fluxcharge.reduction import tidy_hamiltonian

    res = library.zero_pi().hamiltonian(ground="v1", strict=False, canonical=True)
    # the collected form is much shorter than the raw expanded sum...
    assert len(sp.Add.make_args(res.H)) < len(sp.Add.make_args(sp.expand(res.H)))
    # ...but it is the *same* Hamiltonian
    coords = [c for a, b, _ in res.conjugate_pairs for c in (a, b)]
    assert sp.expand(res.H - tidy_hamiltonian(sp.expand(res.H), coords)) == 0

    # dual of the transmon: -2 E_J cos^2(q/2) + E_J must fold to -E_J cos(q)
    t = Circuit()
    t.add_josephson("e1", "v1", "v2", EJ="E_J")
    t.add_capacitor("e2", "v1", "v2", C="C")
    t.add_loop("f1", ["+e1", "-e2"])
    dr = dual(t).hamiltonian(strict=False, canonical=True)
    cos_args = [c.args[0] for c in dr.H.atoms(sp.cos)]
    assert all(a.as_poly(*a.free_symbols).total_degree() <= 1 for a in cos_args)
    # integer coefficient -> diagonalizes with no explicit cutoff needed
    ev = np.sort(dr.eigenenergies({"E_J": 10.0, "C": 1.0}, n_levels=4))
    assert np.all(np.isreal(ev))


def test_tidy_skips_trig_fold_when_no_powers():
    """The trig fold only runs when there is a cos/sin power or product to
    reduce.  Dualizing the 0-pi qubit gives cos(q1-q2)+cos(q1+q2): already
    linear, so the (very slow) Fu product-to-sum search must be skipped -- the
    reduction has to finish quickly rather than hang."""
    import sympy as sp
    from fluxcharge import library, dual
    from fluxcharge.reduction import _has_trig_power

    q1, q2 = sp.symbols("q1 q2")
    assert _has_trig_power(sp.cos(q1 / 2) ** 2) is True            # power
    assert _has_trig_power(sp.cos(q1) * sp.cos(q2)) is True        # product
    assert _has_trig_power(sp.cos(q1 - q2) + sp.cos(q1 + q2)) is False  # linear

    import time
    t0 = time.time()
    r = dual(library.zero_pi()).hamiltonian(strict=False, canonical=True)
    assert time.time() - t0 < 20.0          # was >60s before the guard
    # cosines stayed separate -- not fused into cos*cos products by Fu
    assert len(r.H.atoms(sp.cos)) >= 2 and _has_trig_power(r.H) is False


def test_schematic_infers_loops_for_planar_layout():
    """A circuit built without declared loops still draws with the crossing-free
    planar layout: schematic() infers faces just like hamiltonian() does."""
    import matplotlib
    matplotlib.use("Agg")
    from fluxcharge import library

    zp = library.zero_pi()
    assert not zp._loops                      # nothing declared
    zp.schematic()                            # should not fall back to spring
    assert zp._loops                          # faces were inferred
    assert zp._planar is True


def test_wavefunction_representation_toggle():
    """The wavefunction plot can be shown in the flux or the charge
    representation (the latter via Fourier transform / charge distribution)."""
    _require_numpy()
    import matplotlib
    matplotlib.use("Agg")
    from fluxcharge import library

    # fluxonium: potential lives in flux; charge view is the FT of the states
    fx = library.fluxonium().hamiltonian(ground="v1", open_loops="f3", canonical=True)
    p = {"E_J": 5.0, "L": 1.0, "C": 1.0, "phi_ext_f1": 0.0}
    ax_phi = fx.plot_potential_wavefunctions(p, n_levels=3, cutoffs={"phi_v2": 80},
                                             representation="flux")
    assert "phi" in ax_phi.get_xlabel()
    ax_q = fx.plot_potential_wavefunctions(p, n_levels=3, cutoffs={"phi_v2": 80},
                                           representation="charge")
    assert "q_" in ax_q.get_xlabel()

    # transmon: charge view is the discrete Cooper-pair-number distribution
    tr = library.transmon().hamiltonian(ground="v1")
    ax_n = tr.plot_potential_wavefunctions({"E_J": 15.0, "C": 1.0}, n_levels=3,
                                           cutoffs={"q_f1": 61}, representation="charge")
    assert "q_" in ax_n.get_xlabel()


def test_sweep_quantities():
    """plot_spectrum supports levels / transitions / anharmonicity; the
    transmon anharmonicity tends to -E_C."""
    _require_numpy()
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    from fluxcharge import library

    tr = library.transmon().hamiltonian(ground="v1")
    spec = tr.sweep("E_J", np.array([20.0, 30.0]), {"C": 1.0}, n_levels=3,
                    cutoffs={"q_f1": 61})
    anh = (spec[:, 2] - spec[:, 1]) - (spec[:, 1] - spec[:, 0])
    assert np.all(anh < 0)                       # negative (transmon)
    assert abs(anh[-1] + 1.0 / 8.0) < 0.02       # -> -E_C = -1/(8C) = -0.125

    for q in ("levels", "transitions", "anharmonicity"):
        ax = tr.plot_spectrum("E_J", np.linspace(5, 30, 6), {"C": 1.0},
                              n_levels=3, cutoffs={"q_f1": 61}, quantity=q)
        assert ax is not None
    # anharmonicity needs >= 3 levels
    with pytest.raises(ValueError):
        tr.plot_spectrum("E_J", np.linspace(5, 30, 6), {"C": 1.0},
                         n_levels=2, cutoffs={"q_f1": 61}, quantity="anharmonicity")


def test_to_qutip_matches_and_supports_gyrator():
    """to_qutip exports a Qobj Hamiltonian whose spectrum matches fluxcharge,
    including a gyrator circuit (which scqubits cannot represent)."""
    try:
        import qutip  # noqa: F401
    except ImportError:
        if hasattr(pytest, "skip"):
            pytest.skip("qutip not installed")
        return
    _require_numpy()
    import numpy as np
    from fluxcharge import library

    tr = library.transmon().hamiltonian(ground="v1")
    p = {"E_J": 15.0, "C": 1.0}
    m = tr.to_qutip(p, cutoffs={"q_f1": 61})
    assert m["H"].isherm and "q_f1" in m["operators"]
    ev_q = np.sort(m["H"].eigenenergies())[:5]
    ev_f = tr.eigenenergies(p, n_levels=5, cutoffs={"q_f1": 61})
    assert np.allclose(ev_q - ev_q[0], ev_f - ev_f[0], atol=1e-9)

    # the gyrator circulator exports too (scqubits has no phi*q cross term)
    cir = library.circulator().hamiltonian(ground="v1", open_loops="f4", canonical=True)
    pc = {"E_J": 10.0, "C": 1.0, "G": 0.5}
    mc = cir.to_qutip(pc, cutoffs={"phi_v2": 40})
    ev_qc = np.sort(mc["H"].eigenenergies())[:5]
    ev_fc = cir.eigenenergies(pc, n_levels=5, cutoffs={"phi_v2": 40})
    assert np.allclose(ev_qc - ev_qc[0], ev_fc - ev_fc[0], atol=1e-6)


def test_library_phase_slip_is_transmon_dual():
    """The phase-slip qubit (flux basis) and the transmon (charge basis) share a
    spectrum under E_S<->E_J, L<->C -- the LCG duality, from the library."""
    _require_numpy()
    import numpy as np
    from fluxcharge import library
    tr = library.transmon().hamiltonian(ground="v1")
    ev_t = tr.eigenenergies({"E_J": 10.0, "C": 1.0}, n_levels=5, cutoffs={"q_f1": 81})
    qp = library.phase_slip_qubit(charge_bias=False).hamiltonian(ground="v1")
    ev_q = qp.eigenenergies({"E_S": 10.0, "L": 1.0}, n_levels=5, cutoffs={"phi_v2": 80})
    assert np.allclose(ev_t, ev_q, atol=1e-9)


def test_matrix_elements_and_t1():
    """Charge matrix elements of the transmon, and the golden-rule T1 identity."""
    _require_numpy()
    import numpy as np
    ckt = Circuit()
    ckt.add_josephson("e1", "v1", "v2", EJ="E_J")
    ckt.add_capacitor("e2", "v1", "v2", C="C")
    ckt.add_loop("f1", ["+e1", "-e2"])
    res = ckt.hamiltonian(ground="v1")
    p = ckt.natural_params({"C": "70fF", "E_J": "15GHz"})
    M = res.matrix_elements("q_f1", p, n_levels=4, cutoffs={"q_f1": 81})
    assert M.shape == (4, 4)
    assert abs(M[0, 0]) < 1e-9                 # no diagonal charge element (parity)
    assert abs(M[0, 1]) > 0.5                  # nonzero 0<->1 charge element
    assert np.allclose(M, M.conj().T, atol=1e-9)   # Hermitian
    # golden rule with flat S: rate = |<0|n|1>|^2 (S(w)+S(-w)) = 2|<0|n|1>|^2
    _, rate = res.t1(p, "q_f1", lambda w: 1.0, cutoffs={"q_f1": 81})
    assert abs(rate - 2 * abs(M[0, 1]) ** 2) < 1e-6


def test_flux_sweet_spot_sensitivity():
    """The 0->1 transition is first-order insensitive to flux at the fluxonium
    sweet spot (phi_ext = pi) and sensitive away from it."""
    _require_numpy()
    import math
    ckt = Circuit()
    ckt.add_josephson("e1", "v1", "v2", EJ="E_J")
    ckt.add_inductor("e2", "v1", "v2", L="L")
    ckt.add_capacitor("e3", "v1", "v2", C="C")
    ckt.add_loop("f1", ["+e1", "-e2"]); ckt.add_loop("f2", ["+e2", "-e3"])
    ckt.add_loop("f3", ["-e1", "+e3"])
    ckt.set_flux_bias("f1")
    res = ckt.hamiltonian(ground="v1", open_loops="f3")
    base = ckt.natural_params({"E_J": "4GHz", "C": "1GHz", "L": "1GHz"})
    at_sweet = dict(base); at_sweet["phi_ext_f1"] = math.pi
    df_s, d2f_s = res.transition_sensitivity("phi_ext_f1", at_sweet,
                                             cutoffs={"phi_v2": 60})
    away = dict(base); away["phi_ext_f1"] = 0.3
    df_a, _ = res.transition_sensitivity("phi_ext_f1", away, cutoffs={"phi_v2": 60})
    assert abs(df_s) < 1e-3            # sweet spot: first-order insensitive
    assert abs(df_a) > 0.02           # away: sensitive
    assert abs(d2f_s) > 1.0           # curvature dominates at the sweet spot


def test_physical_units_transmon():
    """Physical units: C in fF + E_J in GHz give the transmon frequency in GHz,
    matching sqrt(8 E_J E_C) - E_C, with anharmonicity near -E_C."""
    _require_numpy()
    import numpy as np
    import math
    from fluxcharge import charging_energy_GHz
    ckt = Circuit()
    ckt.add_josephson("e1", "v1", "v2", EJ="E_J")
    ckt.add_capacitor("e2", "v1", "v2", C="C")
    ckt.add_loop("f1", ["+e1", "-e2"])
    res = ckt.hamiltonian(ground="v1")
    p = ckt.natural_params({"C": "70fF", "E_J": "15GHz"})
    ev = res.eigenenergies(p, n_levels=3, cutoffs={"q_f1": 81})
    E_C = charging_energy_GHz(70.0)
    f01 = ev[1] - ev[0]
    assert abs(f01 - (math.sqrt(8 * 15 * E_C) - E_C)) < 0.05   # GHz
    anharm = (ev[2] - ev[1]) - (ev[1] - ev[0])
    assert abs(anharm + E_C) < 0.05                            # anharm ~ -E_C
    assert abs(charging_energy_GHz(70.0) - 19.37 / 70.0) < 1e-3


def test_units_parse_quantity():
    from fluxcharge.units import parse_quantity
    assert parse_quantity("70fF") == (70.0, "C")
    assert parse_quantity("150nH") == (150.0, "L")
    assert parse_quantity("15GHz") == (15.0, "F")
    assert parse_quantity((1.0, "uH")) == (1000.0, "L")     # -> nH
    assert parse_quantity(0.5) == (0.5, None)


def test_offset_charge_transmon():
    """An offset charge on the island gives H = (q - n_g)^2/2C - E_J cos(phi)."""
    ckt = Circuit()
    ckt.add_josephson("e1", "v1", "v2", EJ="E_J")
    ckt.add_capacitor("e2", "v1", "v2", C="C")
    ckt.add_loop("f1", ["+e1", "-e2"])
    ng = ckt.set_offset_charge("v2")
    res = ckt.hamiltonian(ground="v1", canonical=True)
    C, EJ = sp.Symbol("C"), sp.Symbol("E_J")
    phi, q = sp.Symbol("phi_v2"), sp.Symbol("q_f1")
    assert sp.simplify(res.H - ((q - ng) ** 2 / (2 * C) - EJ * sp.cos(phi))) == 0
    assert ng in ckt.parameters


def test_external_flux_fluxonium():
    """An external flux through the loop is 2*pi-periodic with the half-flux
    sweet spot at pi (the symbol equals the physical loop flux)."""
    _require_numpy()
    import numpy as np
    ckt = Circuit()
    ckt.add_josephson("e1", "v1", "v2", EJ="E_J")
    ckt.add_inductor("e2", "v1", "v2", L="L")
    ckt.add_capacitor("e3", "v1", "v2", C="C")
    ckt.add_loop("f1", ["+e1", "-e2"]); ckt.add_loop("f2", ["+e2", "-e3"])
    ckt.add_loop("f3", ["-e1", "+e3"])
    px = ckt.set_flux_bias("f1")
    res = ckt.hamiltonian(ground="v1", open_loops="f3", canonical=True)

    def e01(val):
        ev = res.eigenenergies({"E_J": 4.0, "L": 1.0, "C": 1.0, str(px): val},
                               n_levels=2, cutoffs={"phi_v2": 60})
        return ev[1] - ev[0]
    import math
    assert abs(e01(0.0) - e01(2 * math.pi)) < 1e-3      # period 2*pi
    assert e01(math.pi) < 0.05                            # sweet-spot gap collapse
    assert e01(0.0) > 1.0                                 # away from the sweet spot


def test_bias_netlist_round_trip():
    """flux / offset directives parse and round-trip through to_netlist."""
    from fluxcharge import from_netlist, to_netlist
    text = ("J e1 v1 v2 E_J\nL e2 v1 v2 L\nC e3 v1 v2 C\n"
            "loop f1 +e1 -e2\nloop f2 +e2 -e3\nloop f3 -e1 +e3\n"
            "ground v1\nopen f3\nflux f1 phi_ext\noffset v2 n_g\n")
    ckt = from_netlist(text)
    assert "f1" in ckt._flux_bias and "v2" in ckt._offset_charge
    rt = from_netlist(to_netlist(ckt))
    assert str(rt._flux_bias["f1"]) == "phi_ext"
    assert str(rt._offset_charge["v2"]) == "n_g"


def test_dual_infers_loops_when_undeclared():
    """dual() auto-infers the planar faces when no loops are declared (e.g. a
    circuit imported from scqubits, whose netlist has no loop lines)."""
    from fluxcharge import dual
    ckt = Circuit()                       # no add_loop
    ckt.add_josephson("e1", "1", "0", EJ="E_J")
    ckt.add_capacitor("e2", "1", "0", C="C")
    d = dual(ckt)                         # should not raise
    kinds = {type(el).__name__ for el in d._elements}
    assert "QuantumPhaseSlip" in kinds and "Inductor" in kinds   # JJ->QPS, C->L
    assert d.hamiltonian(strict=False, canonical=True).complete


def test_dual_rejects_nonplanar():
    """A non-planar circuit has no faces, so dual() refuses with a clear error."""
    import warnings
    from fluxcharge import dual
    c = Circuit()
    nodes = ["v1", "v2", "v3", "v4", "v5"]
    k = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            k += 1
            c.add_inductor(f"e{k}", nodes[i], nodes[j], L=f"L{k}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            dual(c)


def test_dual_fluxonium_spectrum_and_wavefunction_plot():
    """The dual of the fluxonium has the same spectrum, and its wavefunctions
    plot against the *charge* (the phase-slip dual puts cos on the charge, not a
    gyrator cross term)."""
    _require_numpy()
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    from fluxcharge import library, dual

    fx = library.fluxonium()
    rf = fx.hamiltonian(ground="v1", open_loops="f3", canonical=True)
    rd = dual(fx).hamiltonian(strict=False, canonical=True)
    p = {"E_J": 5.0, "L": 1.0, "C": 1.0, "phi_ext_f1": 0.0}
    fc = [str(b) for _a, b, _c in rf.conjugate_pairs][0]
    dc = [str(b) for _a, b, _c in rd.conjugate_pairs][0]
    ev_f = rf.eigenenergies(p, n_levels=6, cutoffs={fc: 90}); ev_f -= ev_f[0]
    ev_d = rd.eigenenergies(p, n_levels=6, cutoffs={dc: 90}); ev_d -= ev_d[0]
    assert np.allclose(ev_f, ev_d, atol=1e-6)        # duality preserves the spectrum

    # the dual plots against the charge (cos of charge), not as a gyrator failure
    ax = rd.plot_potential_wavefunctions(p, n_levels=4, cutoffs={dc: 90})
    assert "q_" in ax.get_xlabel()


def test_dual_carries_bias():
    """Under the LCG dual, an offset charge on a node becomes an external flux
    through the dual loop (and vice versa)."""
    from fluxcharge import dual
    t = Circuit()
    t.add_josephson("e1", "v1", "v2", EJ="E_J")
    t.add_capacitor("e2", "v1", "v2", C="C")
    t.add_loop("f1", ["+e1", "-e2"])
    t.set_offset_charge("v2", "n_g")
    d = dual(t)
    # node v2 -> dual loop v2 carries the flux bias
    assert "v2" in d._flux_bias and str(d._flux_bias["v2"]) == "n_g"
    assert not d._offset_charge


def test_gui_qol_helpers(tmp_path, monkeypatch):
    """Clipboard text, CSV export, error-line parsing and session round-trip."""
    from fluxcharge.gui import (compute, hamiltonian_clipboard, eigenenergies_csv,
                                netlist_error_line, load_session, save_session)
    out = compute("J e1 v1 v2 E_J\nC e2 v1 v2 C\nloop f1 +e1 -e2\nground v1",
                  draw=False)
    # clipboard formats
    assert hamiltonian_clipboard(out, "latex").startswith(r"\hat{H} = ")
    assert "E_J" in hamiltonian_clipboard(out, "sympy")
    assert "4 E_{C}" in hamiltonian_clipboard(out, "latex", energy=True)
    assert "phi" in hamiltonian_clipboard(out, "commutators")
    # csv
    csv = eigenenergies_csv([1.0, 2.5, 3.0], modes=[("phi_v2", "q_f1", "periodic")])
    assert "level,energy" in csv and "0,1" in csv and "periodic" in csv
    # error-line parsing
    assert netlist_error_line("line 7: bad token") == 7
    assert netlist_error_line("no number here") is None
    # session round-trip (redirect config dir to a temp path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert load_session() == {}
    assert save_session({"netlist": "J e1 v1 v2", "levels": 5})
    got = load_session()
    assert got["netlist"] == "J e1 v1 v2" and got["levels"] == 5


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

"""
General symplectic (Darboux) canonicalization and compactness classification
for multi-mode circuits, reciprocal and non-reciprocal.

This implements the linear-algebra core of the pipeline described in the project
notes (``NONRECIPROCAL_QUANTIZATION.md`` / ``zeropi_canonicalization.pdf``):

* :func:`canonical_from_bracket` -- bring the reduced bracket matrix
  ``Pi = f^{-1}`` to the standard symplectic form ``J`` by a real linear map.
  This replaces the per-graph-object "one charge per flux" shortcut, which is
  correct only when ``Pi`` is already block-antidiagonal-diagonal (every
  single-mode circuit) and silently wrong for a genuinely multi-mode circuit,
  where the flux<->charge block is dense and gyrators populate the flux-flux and
  charge-charge blocks.
* :func:`symplectic_eigenvalues` -- the Williamson normal-mode frequencies of a
  quadratic system, a convention-free spectral oracle invariant under any linear
  canonical map.
* :func:`integer_kernel` -- the integer kernel used to build the gauge
  (compactness) lattice from a circuit's linear-inductor incidence / linear-
  capacitor loop matrix.

The numeric routines (``canonical_from_bracket``, ``symplectic_eigenvalues``)
require numpy/scipy; they are only invoked from the numeric layer, after circuit
parameters have been substituted to floats.  The symbolic Hamiltonian is never
disturbed by these.
"""

from __future__ import annotations

from typing import List


class CompactLatticeError(ValueError):
    """A real symplectic canonicalization rotated a compact (periodic)
    coordinate into a non-lattice combination, so no clean integer/periodic
    product basis exists (an ``Sp(2n, Z)`` obstruction).

    Raised rather than silently substituting an oscillator basis for a compact
    mode -- the dual of the 0-pi qubit is the canonical place this can occur.
    """


def _standard_J(n: int):
    import numpy as np
    return np.block([[np.zeros((n, n)), np.eye(n)],
                     [-np.eye(n), np.zeros((n, n))]])


def canonical_from_bracket(Pi, tol: float = 1e-9):
    """Return ``T`` (real, ``2n x 2n``) with ``T @ Pi @ T.T == J``.

    With the reduced **bracket matrix** ``Pi = f^{-1}`` (so the reduced
    coordinates obey ``[xi_i, xi_j] = i*hbar*Pi_ij``), the canonical coordinates
    ``eta = T @ xi`` obey ``[eta_i, eta_j] = i*hbar*J_ij`` with
    ``J = [[0, I], [-I, 0]]`` -- i.e. ``{x_i, p_j} = +delta_ij``.

    ``Pi`` must be real, antisymmetric and non-degenerate.  Works for any
    multi-mode circuit, including non-reciprocal ones whose ``Pi`` has nonzero
    flux-flux / charge-charge blocks (so the charge-only fast-path is invalid).
    """
    import numpy as np
    from scipy.linalg import schur

    Pi = np.asarray(Pi, dtype=float)
    n2 = Pi.shape[0]
    if n2 % 2 != 0:
        raise ValueError("bracket matrix must be even-dimensional")
    n = n2 // 2
    if not np.allclose(Pi, -Pi.T, atol=1e-8):
        raise ValueError("bracket matrix must be antisymmetric")

    # Real Schur form: Pi = Z D Z^T, D block-diagonal with 2x2 blocks
    # [[0, b], [-b, 0]] (b real, nonzero iff Pi is non-degenerate), Z orthogonal.
    D, Z = schur(Pi, output="real")
    xrows: List = []
    prows: List = []
    i = 0
    while i < n2:
        b = D[i, i + 1]
        if abs(b) < tol:
            raise ValueError(
                "bracket matrix is degenerate (zero symplectic eigenvalue); "
                "the reduction left an unremoved null direction")
        s = 1.0 / np.sqrt(abs(b))
        ri, ri1 = s * Z[:, i], s * Z[:, i + 1]
        # orient so that {x, p} = +1 (swap the pair when b < 0)
        if b > 0:
            xrows.append(ri)
            prows.append(ri1)
        else:
            xrows.append(ri1)
            prows.append(ri)
        i += 2

    T = np.vstack(xrows + prows)
    J = _standard_J(n)
    if not np.allclose(T @ Pi @ T.T, J, atol=1e-7):
        raise AssertionError("canonicalization failed: T @ Pi @ T.T != J")
    return T


def symplectic_eigenvalues(Pi, K, tol: float = 1e-9):
    """Williamson normal-mode frequencies ``{omega_k}`` of a quadratic system.

    ``Pi = f^{-1}`` is the reduced bracket matrix and ``K`` the Hessian of the
    quadratic energy.  The eigenvalues of ``Pi @ K`` come in conjugate pairs
    ``+/- i*omega_k``; the ``omega_k`` are invariant under any linear canonical
    transformation, so they are a convention-free oracle for the spectrum of a
    purely quadratic Hamiltonian (its level spacings are sums of the
    ``omega_k``).  Returns the ``n`` positive frequencies, sorted.
    """
    import numpy as np
    Pi = np.asarray(Pi, dtype=float)
    K = np.asarray(K, dtype=float)
    w = np.linalg.eigvals(Pi @ K)
    om = np.sort(w.imag[w.imag > tol])
    return om


def integer_kernel(M):
    """An integer basis of ``{x in Z^cols : M x = 0}``.

    Used to build the gauge (compactness) lattice: the flux-shift sublattice is
    the integer kernel of the linear-inductor incidence, the charge-shift
    sublattice the integer kernel of the linear-capacitor loop matrix.  Each
    returned vector is cleared of denominators so its entries are integers.
    """
    import sympy as sp
    M = sp.Matrix(M)
    out: List[List[int]] = []
    for v in M.nullspace():
        rats = [sp.nsimplify(t) for t in v]
        denoms = [sp.Rational(r).q for r in rats]
        d = sp.ilcm(*denoms) if denoms else 1
        vec = [int(d * sp.Rational(r)) for r in rats]
        nz = [abs(x) for x in vec if x != 0]
        g = (nz[0] if len(nz) == 1 else sp.igcd(*nz)) if nz else 1
        out.append([x // g for x in vec])
    return out


def _linear_edge_names(circuit, typename):
    """Edge names whose element is a *linear* L or C (genuine inductor /
    capacitor), excluding Josephson junctions and quantum phase slips -- only
    these contribute a quadratic (periodicity-breaking) energy."""
    return [el._edge.name for el in circuit._elements
            if type(el).__name__ == typename]


def compact_flux_modes(circuit) -> int:
    """Number of compact (periodic) **flux** modes of *circuit*, computed from
    its gauge lattice rather than guessed per graph object.

    A large gauge transformation shifts island phases by 2*pi; it is a symmetry
    of the energy iff it leaves every *linear* inductor flux invariant (the
    Josephson/phase-slip cosines are automatically 2*pi-periodic).  The
    flux-shift sublattice is therefore the integer kernel of the linear-inductor
    incidence: a flux direction is compact iff no linear inductor pins it.
    Removing the global-flux gauge (one per connected component, always a
    symmetry and gauge-fixed by grounding) gives the number of physical compact
    flux modes.

    Reproduces: transmon ``1`` (phi compact), fluxonium ``0`` (phi extended,
    pinned by its inductor), 0-pi ``1`` (theta compact, phi extended).

    The dual statement for compact **charge** modes (the integer kernel of the
    linear-capacitor loop matrix) needs the loop-space gauge bookkeeping that is
    entangled with redundant/inferred loops and parallel edges; it is part of
    the deferred non-reciprocal/dual work and is intentionally not returned here
    rather than returned unverified.
    """
    import sympy as sp
    import networkx as nx

    edges = circuit.edges
    eidx = {e: i for i, e in enumerate(edges)}
    A = circuit.incidence_matrix()        # |E| x |V|
    lin_L = _linear_edge_names(circuit, "Inductor")
    LA = (sp.Matrix([A.row(eidx[e]) for e in lin_L])
          if lin_L else sp.zeros(0, A.cols))

    flux_ker = integer_kernel(LA)
    G = circuit.to_networkx().to_undirected()
    ncomp = nx.number_connected_components(G) if G.number_of_nodes() else 0
    return max(0, len(flux_ker) - ncomp)

"""
Numerical diagonalization of reduced flux-charge Hamiltonians.

This module turns the *symbolic* Hamiltonian produced by
:class:`~fluxcharge.reduction.ReductionResult` into a finite matrix, diagonalizes
it, and returns the spectrum and eigenstates.  It is deliberately
self-contained (only ``numpy`` is required at run time): the gyrator and
quantum-phase-slip circuits this package targets fall outside the model of
existing tools such as scqubits/QuTiP, so the operators are built natively.

Mode-type detection
-------------------
The choice of basis that *correctly* diagonalizes a degree of freedom is fixed
by how its flux enters the potential.  For each canonically conjugate pair
``(flux, charge)`` (read off the reduced symplectic form) we classify the mode
as one of:

* ``EXTENDED`` -- the flux is confined by an inductor, so it carries a
  quadratic term ``~ phi**2`` (possibly in addition to a Josephson cosine).
  The flux is a continuous, bound coordinate and the pair is diagonalized in a
  **harmonic-oscillator (Fock) basis** whose zero-point spread is fixed from
  the quadratic coefficients of ``H``.  *LC oscillator, fluxonium, and -- by the
  flux-charge symmetry -- a quantum phase slip shunting an inductor.*

* ``PERIODIC`` -- the only nonlinearity in the flux is a Josephson cosine and
  there is **no** quadratic ``phi**2`` term, so the flux is compact (lives on
  ``S^1``) and its conjugate charge is integer-valued (Cooper-pair number).
  The pair is diagonalized in the **charge basis**, with an optional offset
  charge ``n_g``.  *Transmon / Cooper-pair box.*

* ``FREE`` -- the flux appears in no potential term at all (purely kinetic).
  Carried in the charge basis; only the offset charge has any effect.

This is the same three-way taxonomy used by scqubits (Chitta, Groszkowski &
Koch, *New J. Phys.* 2022), specialized to the flux-charge symmetric setting:
the test keys on the *flux* member of each conjugate pair, so a quantum phase
slip (which puts the *charge* inside the cosine) shunted by an inductor is an
``EXTENDED`` mode whose cosine is evaluated as a matrix function -- and, by
duality, has exactly the spectrum of the corresponding transmon.

Honesty about assumptions
-------------------------
Three steps are physical choices, surfaced rather than hidden:

* **Basis cutoffs** (oscillator levels / charge range) are truncations; converge
  them and check the spectrum is stable.
* **Operator ordering** of any gyrator cross term (e.g. ``G*phi*q``) is taken in
  the Hermitian Weyl-symmetrized form ``(phi*q + q*phi)/2``; the assembled matrix
  is Hermitized and a warning is raised if a larger-than-numerical
  anti-Hermitian part remains (an ordering ambiguity beyond the bilinear case).
* **S^1 compactness** of a ``PERIODIC`` mode is a modelling choice (manuscript,
  Sec. "Constraints and Quantization"); it follows here from the flux appearing
  only inside a cosine, and can be overridden with ``mode_types=``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import sympy as sp

EXTENDED = "extended"
PERIODIC = "periodic"
DUAL_PERIODIC = "dual-periodic"
FREE = "free"

Number = Union[int, float, complex]


# ----------------------------------------------------------------------
# mode classification
# ----------------------------------------------------------------------
@dataclass
class Mode:
    """One reduced degree of freedom and how it will be diagonalized.

    Attributes
    ----------
    flux, charge : sympy.Symbol
        The conjugate pair, with ``flux`` the ``phi_*`` member and ``charge``
        the ``q_*`` member.
    coeff : sympy.Expr
        Symplectic coefficient of the pair (``+/-1`` after canonicalization).
    kind : str
        ``EXTENDED``, ``PERIODIC`` or ``FREE``.
    """

    flux: sp.Symbol
    charge: sp.Symbol
    coeff: sp.Expr
    kind: str

    @property
    def discrete(self) -> sp.Symbol:
        """The variable carried in an integer basis (kinetic / quadratic member).

        For a ``PERIODIC`` (transmon-like) mode this is the charge; for a
        ``DUAL_PERIODIC`` (quantum-phase-slip-like) mode it is the flux.
        Undefined for ``EXTENDED``/``FREE`` modes.
        """
        return self.charge if self.kind in (PERIODIC, FREE) else self.flux

    @property
    def compact(self) -> sp.Symbol:
        """The compact (``S^1``) variable sitting inside the cosine."""
        return self.flux if self.kind in (PERIODIC, FREE) else self.charge

    def __repr__(self):
        return f"Mode({self.flux}, {self.charge}; {self.kind})"


def _is_flux(sym: sp.Symbol) -> bool:
    return str(sym).startswith("phi_") or str(sym).startswith("\\phi")


def _split_pair(a: sp.Symbol, b: sp.Symbol) -> Tuple[sp.Symbol, sp.Symbol]:
    """Return ``(flux, charge)`` from a conjugate pair in either order."""
    if _is_flux(a) and not _is_flux(b):
        return a, b
    if _is_flux(b) and not _is_flux(a):
        return b, a
    # ambiguous (e.g. both charges after a dual): treat the first as "flux"
    return a, b


def _has_quadratic(H: sp.Expr, sym: sp.Symbol) -> bool:
    return sp.expand(H).coeff(sym, 2) != 0


def _inside_cos_or_sin(H: sp.Expr, sym: sp.Symbol) -> bool:
    for fn in (sp.cos, sp.sin):
        for atom in H.atoms(fn):
            if sym in atom.args[0].free_symbols:
                return True
    return False


def _polynomial_part(H: sp.Expr) -> sp.Expr:
    """``H`` with the Josephson/phase-slip cosine/sine terms removed -- i.e. the
    inductive/capacitive quadratic energy that sets confinement."""
    return sp.Add(*[t for t in sp.Add.make_args(sp.expand(H))
                    if not t.has(sp.cos, sp.sin)])


def check_compact_frame(result, modes, Hnum):
    """Guard against a compact mode hidden by coordinate mixing.

    The per-pair classification (:func:`classify_modes`) decides extended vs
    compact by looking at each coordinate in isolation.  That is correct only
    when an unconfined direction coincides with a coordinate.  For a multi-mode
    circuit the inductive (resp. capacitive) quadratic form can be **degenerate
    along a linear combination** -- 0-pi's ``theta = phi_n2 + phi_n3`` carries no
    inductive energy, yet every node flux individually does, so the eye-rule
    wrongly calls all modes extended and the spectrum diverges.

    Here we compare the number of unconfined directions (rank deficiency of the
    numeric quadratic form) against the number the per-coordinate rule actually
    captured.  If the former exceeds the latter there is a compact mode that the
    current coordinate frame does not expose -- and the naive choice generically
    produces a non-integer cosine (``cos(theta/2)``), which has no integer-lattice
    representation.  Rather than return a silently-wrong spectrum we raise
    :class:`~fluxcharge.canonicalize.CompactLatticeError`, asking the user to
    supply a lattice-aware frame (integer cosines on the compact directions).
    """
    import numpy as np
    from .canonicalize import CompactLatticeError

    poly = _polynomial_part(Hnum)
    fluxes = [m.flux for m in modes]
    charges = [m.charge for m in modes]

    def _rank(coords):
        if not coords:
            return 0
        K = sp.hessian(poly, coords)
        K = np.array(K.tolist(), dtype=complex).real.astype(float)
        return int(np.linalg.matrix_rank(K, tol=1e-9)) if K.size else 0

    # unconfined = rank-deficiency of the quadratic form;
    # coordinate-captured = coordinates that individually lack a quadratic term.
    for label, coords, captured_kinds in (
            ("flux", fluxes, (PERIODIC, FREE)),
            ("charge", charges, (DUAL_PERIODIC, FREE))):
        n_unconfined = len(coords) - _rank(coords)
        n_coord = sum(1 for c in coords if not _has_quadratic(poly, c))
        n_captured = sum(1 for m in modes if m.kind in captured_kinds)
        # a direction is hidden if more directions are unconfined than the
        # number of coordinates that are individually unconfined and tagged
        if n_unconfined > n_coord:
            raise CompactLatticeError(
                f"this circuit has {n_unconfined} unconfined {label} direction(s) "
                f"but only {n_coord} lie along a coordinate, so a compact mode is a "
                "linear combination hidden by the frame (e.g. 0-pi's theta = "
                "phi_n2 + phi_n3). The natural choice yields a half-integer cosine "
                "cos(x/2) with no integer-lattice representation. Supply a "
                "lattice-aware frame in which the compact direction is a coordinate "
                "with integer cosines (pass mode_types=/a coordinate frame), rather "
                "than relying on automatic classification.")


def classify_modes(result, mode_types: Optional[Dict] = None) -> List[Mode]:
    """Classify every conjugate pair of *result* into a :class:`Mode`.

    The classification is symmetric in flux and charge.  With ``p2``/``q2`` the
    presence of a quadratic ``phi**2``/``q**2`` term and ``pcos``/``qcos`` the
    appearance of the flux/charge inside a cosine:

    * both quadratics present -> ``EXTENDED`` (oscillator basis);
    * ``q2`` and ``pcos`` but no ``p2`` -> ``PERIODIC`` (charge basis, flux
      compact) -- the transmon;
    * ``p2`` and ``qcos`` but no ``q2`` -> ``DUAL_PERIODIC`` (flux basis, charge
      compact) -- a quantum phase slip shunting an inductor, the transmon's dual;
    * otherwise ``FREE``.

    Pass ``mode_types={flux_symbol: "extended"|"periodic"|"dual-periodic"|"free"}``
    to override.
    """
    H = sp.expand(result.H)
    overrides = {sp.sympify(k): v for k, v in (mode_types or {}).items()}
    modes: List[Mode] = []
    for a, b, c in result.conjugate_pairs:
        flux, charge = _split_pair(a, b)
        p2, q2 = _has_quadratic(H, flux), _has_quadratic(H, charge)
        pcos, qcos = _inside_cos_or_sin(H, flux), _inside_cos_or_sin(H, charge)
        if flux in overrides:
            kind = overrides[flux]
        elif p2 and q2:
            kind = EXTENDED
        elif q2 and pcos:
            kind = PERIODIC
        elif p2 and qcos:
            kind = DUAL_PERIODIC
        elif pcos:
            kind = PERIODIC
        elif qcos:
            kind = DUAL_PERIODIC
        else:
            kind = FREE
        modes.append(Mode(flux=flux, charge=charge, coeff=c, kind=kind))
    return modes


# ----------------------------------------------------------------------
# small linear-algebra helpers (numpy, imported lazily)
# ----------------------------------------------------------------------
def _np():
    import numpy as np
    return np


def _annihilation(n: int):
    np = _np()
    return np.diag(np.sqrt(np.arange(1, n)), 1)


def _expm_herm(M):
    """Matrix exponential ``expm(i*M)`` of a Hermitian ``M`` via eigendecomp."""
    np = _np()
    w, V = np.linalg.eigh(M)
    return (V * np.exp(1j * w)) @ V.conj().T


def _kron_list(mats):
    np = _np()
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


# ----------------------------------------------------------------------
# operator construction
# ----------------------------------------------------------------------
class _OperatorBuilder:
    """Builds full-space operators for the coordinates of a reduced circuit."""

    def __init__(self, result, params, cutoffs, offsets, mode_types):
        np = _np()
        self.np = np
        self.result = result
        self.modes = classify_modes(result, mode_types=mode_types)
        self.params = {sp.sympify(k): complex(v) for k, v in (params or {}).items()}
        self.offsets = {sp.sympify(k): float(v) for k, v in (offsets or {}).items()}

        # numeric Hamiltonian (parameters substituted, coordinates symbolic)
        self.coords = list(result.coordinates)
        missing = (sp.expand(result.H).free_symbols
                   - set(self.coords) - set(self.params))
        if missing:
            raise ValueError(
                "missing numeric values for parameters: "
                + ", ".join(sorted(map(str, missing)))
                + " -- pass them in params=")
        self.Hnum = sp.expand(result.H.subs(self.params))

        # refuse to silently mis-quantize a compact mode hidden by the frame
        check_compact_frame(result, self.modes, self.Hnum)

        # per-mode local dimension
        # default basis size per mode, scaled down for many modes so the
        # tensor-product dimension stays tractable (a single mode keeps 30/31;
        # 3 modes -> ~16 each, etc.).  Override per mode with cutoffs=.
        base = {EXTENDED: 30, PERIODIC: 31, DUAL_PERIODIC: 31, FREE: 31}
        n = max(1, len(self.modes))
        # cap the per-mode default so the tensor dimension (hence the matrix
        # build, which does an expm per cosine) stays responsive for multi-mode
        # circuits; the user raises cutoffs= deliberately to converge.
        cap = max(7, int(round(1000 ** (1.0 / n))))
        self.dims: List[int] = []
        self.cut = cutoffs or {}
        for m in self.modes:
            d = self.cut.get(str(m.flux), self.cut.get(m.flux))
            if d is None:
                d = self.cut.get(str(m.charge), self.cut.get(m.charge))
            d = int(d) if d is not None else min(base[m.kind], cap)
            if m.kind != EXTENDED:        # integer basis must be odd (-ncut..ncut)
                d = 2 * ((d - 1) // 2) + 1
            self.dims.append(d)

        self._build_operators()

    def _quad(self, sym: sp.Symbol) -> float:
        c = self.Hnum.coeff(sym, 2)
        if c.free_symbols:
            raise ValueError(f"quadratic coefficient of {sym} is not numeric: {c}")
        return float(sp.re(complex(c)))

    def _embed_one(self, idx, local):
        np = self.np
        mats = [np.eye(d) if j != idx else local for j, d in enumerate(self.dims)]
        return _kron_list(mats)

    def _build_operators(self):
        """Full-space operators, keyed by coordinate symbol.

        ``self.op`` holds a bare (Hermitian) matrix for every coordinate that
        has one -- both members of an ``EXTENDED`` pair, and the *discrete*
        member of a ``PERIODIC``/``DUAL_PERIODIC``/``FREE`` mode.  ``self.shift``
        holds, for each compact coordinate ``x``, the operator ``e^{+i x}`` that
        raises its conjugate integer by one.
        """
        np = self.np
        self.dim_total = int(np.prod(self.dims)) if self.dims else 1
        self.op: Dict[sp.Symbol, object] = {}
        self.shift: Dict[sp.Symbol, object] = {}
        for i, m in enumerate(self.modes):
            d = self.dims[i]
            if m.kind == EXTENDED:
                aphi, aq = self._quad(m.flux), self._quad(m.charge)
                if not (aphi > 0 and aq > 0):
                    raise NotImplementedError(
                        f"extended mode {m.flux} needs positive quadratic flux and "
                        f"charge coefficients to set an oscillator basis (got phi^2 "
                        f"coeff {aphi}, q^2 coeff {aq}); override with mode_types=.")
                phi_zpf = np.sqrt(0.5) * (aq / aphi) ** 0.25
                q_zpf = np.sqrt(0.5) * (aphi / aq) ** 0.25
                a = _annihilation(d)
                ad = a.conj().T
                self.op[m.flux] = self._embed_one(i, phi_zpf * (a + ad))
                self.op[m.charge] = self._embed_one(i, 1j * q_zpf * (ad - a))
            else:  # PERIODIC / DUAL_PERIODIC / FREE -> integer basis
                ncut = (d - 1) // 2
                n = np.arange(-ncut, ncut + 1).astype(complex)
                disc = m.discrete
                off = self.offsets.get(disc, self.offsets.get(str(disc), 0.0))
                self.op[disc] = self._embed_one(i, np.diag(n + off))
                # e^{+i*compact} raises the discrete integer by one
                S = np.diag(np.ones(len(n) - 1), -1).astype(complex)
                self.shift[m.compact] = self._embed_one(i, S)

    # -- assemble the Hamiltonian matrix -------------------------------
    def _exp_arg(self, arg: sp.Expr):
        """``expm(i*arg)`` as a full-space matrix, for a *linear* ``arg``."""
        np = self.np
        arg = sp.expand(arg)
        coeffs = arg.as_coefficients_dict()
        phase = 0j
        herm = np.zeros((self.dim_total, self.dim_total), dtype=complex)
        E_shift = np.eye(self.dim_total, dtype=complex)
        for sym, c in coeffs.items():
            c = complex(c)
            if not getattr(sym, "free_symbols", None):  # numeric constant term
                phase += c
                continue
            if sym in self.op and sym not in self.shift:
                herm = herm + c * self.op[sym]   # extended / discrete operator
            elif sym in self.shift:
                if abs(c.imag) > 1e-12 or abs(c.real - round(c.real)) > 1e-9:
                    raise NotImplementedError(
                        f"compact coordinate {sym} must enter a cosine with integer "
                        f"coefficient (got {c}); it sets the conjugate-integer shift.")
                k = int(round(c.real))
                S = self.shift[sym]
                E_shift = E_shift @ (np.linalg.matrix_power(S, k) if k >= 0
                                     else np.linalg.matrix_power(S.conj().T, -k))
            else:
                raise NotImplementedError(
                    f"coordinate {sym} is not supported inside a cosine")
        return np.exp(1j * phase) * (_expm_herm(herm) @ E_shift)

    def _factor_matrix(self, factor: sp.Expr):
        np = self.np
        I = np.eye(self.dim_total, dtype=complex)
        if factor.is_number:
            return complex(factor) * I
        if isinstance(factor, sp.cos):
            E = self._exp_arg(factor.args[0])
            return 0.5 * (E + E.conj().T)
        if isinstance(factor, sp.sin):
            E = self._exp_arg(factor.args[0])
            return (E - E.conj().T) / 2j
        if isinstance(factor, sp.Pow):
            base, exp = factor.args
            if not (exp.is_integer and exp >= 0):
                raise NotImplementedError(f"non-integer/negative power: {factor}")
            return np.linalg.matrix_power(self._factor_matrix(base), int(exp))
        if isinstance(factor, sp.Symbol):
            if factor in self.op:
                return self.op[factor]
            raise NotImplementedError(
                f"bare coordinate {factor} has no operator (a compact coordinate "
                f"may only appear inside a cosine)")
        raise NotImplementedError(f"cannot evaluate factor {factor!r}")

    def matrix(self):
        """The (Hermitized) Hamiltonian matrix."""
        np = self.np
        H = np.zeros((self.dim_total, self.dim_total), dtype=complex)
        # macOS's Accelerate BLAS raises spurious fp flags from matmul on arrays
        # with zeros; the results are correct, so silence them and assert finite.
        max_noncommuting = 0
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for term in sp.Add.make_args(self.Hnum):
                scalar = 1.0 + 0j
                opmat = np.eye(self.dim_total, dtype=complex)
                n_ops = 0
                for factor in sp.Mul.make_args(term):
                    if factor.is_number:
                        scalar *= complex(factor)
                    else:
                        opmat = opmat @ self._factor_matrix(factor)
                        n_ops += factor.as_base_exp()[1] if factor.is_Pow else 1
                max_noncommuting = max(max_noncommuting, n_ops)
                H = H + scalar * opmat
        if not np.isfinite(H).all():
            raise FloatingPointError(
                "Hamiltonian matrix contains non-finite entries; check parameters "
                "and basis cutoffs.")
        # Hermitize.  For a Hamiltonian that is at most quadratic in the
        # coordinates (plus exact cosines) -- the generic LCG case -- this is the
        # unique Weyl ordering and is exact, including a gyrator's bilinear cross
        # term (phi*q + q*phi)/2.  A genuine ordering ambiguity only appears for a
        # monomial of degree >= 3 in non-commuting coordinates, which we flag.
        if max_noncommuting >= 3:
            anti = np.linalg.norm(H - H.conj().T)
            if anti > 1e-9 * max(np.linalg.norm(H), 1.0):
                warnings.warn(
                    f"Hamiltonian has a degree-{max_noncommuting} monomial in "
                    "non-commuting coordinates; its operator ordering is ambiguous "
                    "and has been Weyl-symmetrized. Verify against result.H.",
                    stacklevel=2)
        return 0.5 * (H + H.conj().T)


# ----------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------
def hamiltonian_matrix(result, params=None, cutoffs=None, offsets=None,
                       mode_types=None):
    """Return the dense numeric Hamiltonian matrix of a reduced circuit.

    Parameters
    ----------
    result : ReductionResult
        A *complete* reduction; it is canonicalized internally so each conjugate
        pair has unit symplectic coefficient.
    params : dict
        Numeric values for every symbolic parameter in ``result.H``
        (e.g. ``{"E_J": 15.0, "C": 1.0}``).  Keys may be symbols or strings.
    cutoffs : dict, optional
        Per-mode basis size, keyed by the flux or charge symbol (or its name):
        number of oscillator levels for an ``EXTENDED`` mode, or total charge
        states (``2*ncut+1``) for a ``PERIODIC``/``FREE`` mode.
    offsets : dict, optional
        Offset charge ``n_g`` for ``PERIODIC``/``FREE`` modes, keyed by the
        charge symbol.
    mode_types : dict, optional
        Override the automatic classification, keyed by the flux symbol.
    """
    result = result if result.is_canonical else result.canonical()
    if not result.complete:
        raise ValueError(
            "reduction is incomplete; diagonalization needs a complete "
            "Hamiltonian (see ReductionResult.report()).")
    return _OperatorBuilder(result, params, cutoffs, offsets, mode_types).matrix()


def eigensystem(result, params=None, n_levels=6, cutoffs=None, offsets=None,
                mode_types=None):
    """Lowest ``n_levels`` eigenvalues and eigenvectors. Returns ``(evals, evecs)``."""
    np = _np()
    H = hamiltonian_matrix(result, params, cutoffs, offsets, mode_types)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        w, v = np.linalg.eigh(H)
    n_levels = min(n_levels, len(w))
    return w[:n_levels], v[:, :n_levels]


def eigenenergies(result, params=None, n_levels=6, cutoffs=None, offsets=None,
                  mode_types=None):
    """Lowest ``n_levels`` eigenvalues of the reduced Hamiltonian (sorted)."""
    return eigensystem(result, params, n_levels, cutoffs, offsets, mode_types)[0]


def to_qutip(result, params=None, cutoffs=None, offsets=None, mode_types=None):
    """Export the reduced Hamiltonian and mode operators as QuTiP objects.

    Returns a dict with:

    * ``"H"`` -- the Hamiltonian as a ``qutip.Qobj`` (with tensor ``dims`` set
      per mode, so QuTiP's ``ptrace``/composite tools work);
    * ``"operators"`` -- ``{name: Qobj}`` for each mode's charge / flux operator
      and, for a compact (periodic) coordinate ``x``, the displacement
      ``e^{i x}`` (keyed ``"expi_<x>"``);
    * ``"modes"`` -- the ``(flux, charge, kind)`` list;
    * ``"dims"`` -- the per-mode Hilbert-space dimensions.

    Hands fluxcharge's quantization to QuTiP's mature engine -- time evolution,
    Lindblad master equations, expectation values -- which, working from raw
    operator matrices, supports the gyrator and quantum-phase-slip circuits that
    scqubits cannot represent.  Requires ``qutip``.
    """
    try:
        import qutip
    except ImportError as exc:  # pragma: no cover
        raise ImportError("to_qutip needs qutip: pip install qutip") from exc

    result = result if result.is_canonical else result.canonical()
    if not result.complete:
        raise ValueError("reduction is incomplete; cannot export a Hamiltonian.")
    builder = _OperatorBuilder(result, params, cutoffs, offsets, mode_types)
    dims = list(builder.dims)
    qdims = [dims, dims]
    operators = {str(s): qutip.Qobj(M, dims=qdims) for s, M in builder.op.items()}
    for s, S in builder.shift.items():
        operators[f"expi_{s}"] = qutip.Qobj(S, dims=qdims)
    return {
        "H": qutip.Qobj(builder.matrix(), dims=qdims),
        "operators": operators,
        "modes": [(m.flux, m.charge, m.kind) for m in builder.modes],
        "dims": dims,
    }


def sweep(result, parameter, values, params=None, n_levels=6, cutoffs=None,
          offsets=None, mode_types=None, relative=False):
    """Eigenenergies as one parameter is swept.

    *parameter* (a symbol/name) is swept over *values*; it may be a circuit
    parameter (e.g. ``"E_J"``) or an offset charge symbol (e.g. ``"q_f1"``),
    in which case it is routed through ``offsets``.  Returns an array of shape
    ``(len(values), n_levels)``; with ``relative=True`` the ground state is
    subtracted at each point.
    """
    np = _np()
    parameter = sp.sympify(parameter)
    base_params = dict(params or {})
    base_offsets = dict(offsets or {})
    is_offset = any(str(parameter) == str(m.charge)
                    for m in classify_modes(result, mode_types))
    rows = []
    for val in values:
        p, o = dict(base_params), dict(base_offsets)
        (o if is_offset else p)[parameter] = val
        ev = eigenenergies(result, p, n_levels, cutoffs, o, mode_types)
        rows.append(ev - ev[0] if relative else ev)
    return np.array(rows)


"""
Circuit element definitions for the flux-charge symmetric LCG formalism.

Every element lives on one or more oriented edges of the circuit multigraph.
The formalism (Salcedo, Cocquyt, Osborne, Houck) partitions edges into three
mutually exclusive classes:

    C  -- capacitive edges     (energy is a function of edge charge  Q_e)
    I  -- inductive edges      (energy is a function of edge flux     Phi_e)
    G  -- gyrative edges       (store no energy; come in ordered pairs)

For concreteness, following the manuscript, the coherent quantum phase slip is
treated as a (non-linear) *capacitive* element and the Josephson junction as a
(non-linear) *inductive* element.

The convention G_0 = 1 (reference conductance set to unity) is assumed
throughout, so fluxes and charges are dimensionless and interchangeable, and
the reduced flux quantum is set to 1 (phases equal fluxes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import sympy as sp

# Edge classes
CAPACITIVE = "C"
INDUCTIVE = "I"
GYRATIVE = "G"


def _as_symbol(value, default_name) -> sp.Expr:
    """Coerce a user-supplied parameter into a sympy expression.

    Accepts a number, a string (parsed as a symbol/expression) or an existing
    sympy object.  ``None`` produces a fresh positive symbol named *default_name*.
    """
    if value is None:
        return sp.Symbol(default_name, positive=True)
    if isinstance(value, str):
        return sp.sympify(value)
    return sp.sympify(value)


@dataclass
class Edge:
    """A single oriented edge of the circuit graph.

    Attributes
    ----------
    name : str
        Unique label for the edge (e.g. ``"e1"``).
    tail, head : str
        Vertices the edge is directed *from* (tail) and *toward* (head).
        With the incidence convention of the manuscript,
        ``A[e, head] = +1`` and ``A[e, tail] = -1``.
    edge_class : str
        One of :data:`CAPACITIVE`, :data:`INDUCTIVE`, :data:`GYRATIVE`.
    energy : callable or None
        Maps the relevant edge variable (Q_e for capacitive, Phi_e for
        inductive) to the stored energy, as a sympy expression.  ``None`` for
        gyrative edges.
    element : Element
        Back-reference to the owning element.
    """

    name: str
    tail: str
    head: str
    edge_class: str
    energy: Optional[Callable[[sp.Expr], sp.Expr]] = None
    element: "Element" = field(default=None, repr=False)


class Element:
    """Base class for all circuit elements."""

    def edges(self):  # pragma: no cover - interface declaration
        raise NotImplementedError

    @property
    def parameters(self):
        """Return the set of free sympy symbols introduced by this element."""
        syms = set()
        for e in self.edges():
            if e.energy is not None:
                syms |= e.energy(sp.Symbol("_probe")).free_symbols
        return syms - {sp.Symbol("_probe")}


class Capacitor(Element):
    """Linear capacitor: ``E^C(Q) = Q**2 / (2 C)``."""

    def __init__(self, name, tail, head, C=None):
        self.name = name
        self.C = _as_symbol(C, f"C_{{{name}}}")
        self._edge = Edge(
            name=name,
            tail=tail,
            head=head,
            edge_class=CAPACITIVE,
            energy=lambda Q, C=self.C: Q**2 / (2 * C),
            element=self,
        )

    def edges(self):
        return [self._edge]


class Inductor(Element):
    """Linear inductor: ``E^I(Phi) = Phi**2 / (2 L)``."""

    def __init__(self, name, tail, head, L=None):
        self.name = name
        self.L = _as_symbol(L, f"L_{{{name}}}")
        self._edge = Edge(
            name=name,
            tail=tail,
            head=head,
            edge_class=INDUCTIVE,
            energy=lambda Phi, L=self.L: Phi**2 / (2 * L),
            element=self,
        )

    def edges(self):
        return [self._edge]


class JosephsonJunction(Element):
    """Josephson junction, an *inductive* (non-linear) element.

    Stored energy ``E^I(Phi) = -E_J cos(Phi)`` in the reduced-flux-quantum = 1
    convention.  The Lagrangian therefore picks up ``+E_J cos(Phi)``.
    """

    def __init__(self, name, tail, head, EJ=None):
        self.name = name
        self.EJ = _as_symbol(EJ, f"E_J{{{name}}}")
        self._edge = Edge(
            name=name,
            tail=tail,
            head=head,
            edge_class=INDUCTIVE,
            energy=lambda Phi, EJ=self.EJ: -EJ * sp.cos(Phi),
            element=self,
        )

    def edges(self):
        return [self._edge]


class QuantumPhaseSlip(Element):
    """Coherent quantum phase slip, a *capacitive* (non-linear) element.

    Stored energy ``E^C(Q) = -E_S cos(Q)`` (the charge-space dual of the
    Josephson junction), in the convention 2e = 1.
    """

    def __init__(self, name, tail, head, ES=None):
        self.name = name
        self.ES = _as_symbol(ES, f"E_S{{{name}}}")
        self._edge = Edge(
            name=name,
            tail=tail,
            head=head,
            edge_class=CAPACITIVE,
            energy=lambda Q, ES=self.ES: -ES * sp.cos(Q),
            element=self,
        )

    def edges(self):
        return [self._edge]


class Gyrator(Element):
    """Ideal gyrator: an ordered pair of *gyrative* edges (e1, e2).

    Carries a gyration ratio ``G`` (units of conductance).  Reversing the
    ordering of the edge pair flips the sign of ``G``; the package keeps the
    ordering ``(edge1, edge2)`` fixed and exposes ``G`` directly.

    The two half-edges need not lie in the same really-connected component.
    """

    def __init__(self, name1, tail1, head1, name2, tail2, head2, G=None):
        self.name = f"{name1},{name2}"
        self.G = _as_symbol(G, "G")
        self.edge1 = Edge(name1, tail1, head1, GYRATIVE, energy=None, element=self)
        self.edge2 = Edge(name2, tail2, head2, GYRATIVE, energy=None, element=self)

    def edges(self):
        return [self.edge1, self.edge2]

    @property
    def parameters(self):
        return set(self.G.free_symbols)

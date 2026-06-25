"""
A small desktop UI for building a circuit and generating its Hamiltonian.

Run it with::

    fluxcharge-gui
    # or
    python -m fluxcharge.gui

You describe the circuit in the netlist panel (the same format as
:mod:`fluxcharge.netlist`), optionally using the *Add element / gyrator / loop*
helpers so you do not have to type the syntax, then press **Generate**.  The
app draws the schematic and shows the Lagrangian and Hamiltonian.

The UI uses Tkinter, which ships with standard Python on Windows and macOS (on
Linux install the system ``python3-tk`` package).  To turn it into a single
double-clickable executable, freeze it with PyInstaller::

    pyinstaller --onefile -n fluxcharge-gui -c fluxcharge/gui.py
"""

from __future__ import annotations

import json
import os
import re
import tempfile

import sympy as sp

from .elements import Capacitor, Inductor
from .netlist import from_netlist, to_netlist
from .transformations import dual


EXAMPLE = """\
title Circulator
J    e1  v1 v2  E_J
C    e2  v2 v3  C
C    e3  v3 v1  C
gyrator  e4 v1 v3   e5 v2 v3   G
loop  f1  +e3 +e4
loop  f2  +e1 -e4 +e5
loop  f3  +e2 -e5
loop  f4  -e1 -e2 -e3
ground v1
open   f4
"""


def compute(netlist_text, canonical=True, schematic_path=None, draw=True):
    """Headless pipeline used by the UI (and easy to test).

    Parses *netlist_text* and reduces to the Hamiltonian.  When *draw* is true
    (the default) the schematic is also rendered to *schematic_path* (a temp
    file if None); pass ``draw=False`` to skip all matplotlib work -- the UI
    does so to run the (pure-sympy) reduction off the main thread and draw the
    schematic itself on the main thread.  Returns a dict with keys ``circuit``,
    ``schematic`` (path or None), ``H``, ``H_latex``, ``lagrangian``,
    ``report``, ``title`` and the commutator data.  Raises on malformed input.
    """
    ckt = from_netlist(netlist_text)
    ckt.validate()

    ground = getattr(ckt, "ground", None)
    open_loops = getattr(ckt, "open_loops", None) or None
    result = ckt.hamiltonian(ground=ground, open_loops=open_loops, canonical=canonical)

    drawn = None
    if draw:
        if schematic_path is None:
            fd, schematic_path = tempfile.mkstemp(suffix=".png", prefix="fluxcharge_")
            os.close(fd)
        drawn = schematic_path
        try:
            # outer face comes from the planar embedding, independent of the gauge
            ckt.schematic(path=schematic_path)
        except Exception:
            drawn = None

    operators = list(result.coordinates)
    comm = result.commutators()
    comm_latex = r",\ \ ".join(
        f"[{_operator_latex(a)},\\,{_operator_latex(b)}] = {sp.latex(v)}"
        for a, b, v in comm)
    compact = result.compact_coordinates()

    # familiar-units presentation: q -> n, capacitances -> E_C, inductances -> E_L
    capacitances = {el.C for el in ckt._elements if isinstance(el, Capacitor)}
    inductances = {el.L for el in ckt._elements if isinstance(el, Inductor)}
    H_e, comm_e, defs, charge_map = energy_units_form(
        result.H, comm, capacitances, inductances)
    operators_e = [charge_map.get(s, s) for s in operators]
    H_latex_energy = _hamiltonian_latex(H_e, operators_e)
    comm_latex_energy = r",\ \ ".join(
        f"[{_operator_latex(a)},\\,{_operator_latex(b)}] = {sp.latex(v)}"
        for a, b, v in comm_e)
    defs_latex = r",\ \ ".join(sp.latex(d) for d in defs)

    return {
        "title": getattr(ckt, "title", None),
        "circuit": ckt,
        "result": result,
        "schematic": drawn,
        "H": result.H,
        "H_latex": _hamiltonian_latex(result.H, operators),
        "H_energy": H_e,
        "H_latex_energy": H_latex_energy,
        "commutators": comm,
        "commutators_latex": comm_latex,
        "commutators_latex_energy": comm_latex_energy,
        "energy_defs_latex": defs_latex,
        "compact": compact,
        "lagrangian": ckt.lagrangian(),
        "report": result.report(),
    }


def numerical_summary(netlist_text, params, n_levels=6, canonical=True,
                      cutoffs=None, mode_types=None):
    """Headless numerical pipeline used by the UI (and easy to test).

    Parses *netlist_text*, reduces, classifies the modes and diagonalizes with
    the given *params* (a ``{name: value}`` dict).  Returns a dict with keys
    ``modes`` (list of ``(flux, charge, kind)``), ``eigenenergies`` (numpy
    array), ``transitions`` (gaps above the ground state), ``single_mode``
    (bool), and ``result`` (the :class:`ReductionResult`).  Raises on malformed
    input or missing parameters.
    """
    ckt = from_netlist(netlist_text)
    ckt.validate()
    ground = getattr(ckt, "ground", None)
    open_loops = getattr(ckt, "open_loops", None) or None
    result = ckt.hamiltonian(ground=ground, open_loops=open_loops, canonical=canonical)
    return summary_from_result(result, params, n_levels, cutoffs=cutoffs,
                               mode_types=mode_types)


def summary_from_result(result, params, n_levels=6, cutoffs=None, mode_types=None):
    """Build the numerical-summary dict from an already-reduced result.

    Lets the UI skip the (re)reduction when the circuit is unchanged since the
    last Generate -- diagonalization then costs only the matrix build + eigh.
    *mode_types* (``{flux_symbol: "extended"|"periodic"|"dual-periodic"|"free"}``)
    overrides the automatic compact/extended classification per mode.
    """
    modes = result.modes(mode_types=mode_types)
    ev = result.eigenenergies(params, n_levels=n_levels, cutoffs=cutoffs,
                              mode_types=mode_types)
    return {
        "result": result,
        "modes": [(m.flux, m.charge, m.kind) for m in modes],
        "eigenenergies": ev,
        "transitions": [float(ev[i] - ev[0]) for i in range(1, len(ev))],
        "single_mode": len(modes) == 1,
    }


#: human label <-> classify_modes kind, for the mode-type dialog
MODE_KIND_LABELS = [
    ("extended", "Extended  (oscillator / grid)"),
    ("periodic", "Compact flux  (charge basis)"),
    ("dual-periodic", "Compact charge  (flux basis)"),
    ("free", "Free  (no confinement)"),
]


def mode_type_options(result):
    """Per conjugate pair, the data the 'declare mode types' dialog needs:
    ``(flux, charge, default_kind, warning_or_None)``.

    ``default_kind`` is the automatic compact/extended classification.  A warning
    flags modes where the choice is subtle: a coordinate-dependent symplectic
    bracket (gyrator x nonlinearity -- canonical quantization is ambiguous and
    the diagonalizer will likely error), or a free mode (no confinement)."""
    from .numerics import classify_modes, FREE
    modes = classify_modes(result)
    coords = set(result.coordinates)
    H = sp.sympify(result.H)

    # coordinates that enter a cosine/sine NONLINEARLY (the argument's derivative
    # still depends on a coordinate) signal a coordinate-dependent symplectic
    # bracket -- gyrator x nonlinearity -- where canonical quantization is
    # ambiguous and the diagonalizer will likely error.
    bad = set()
    for fn in (sp.cos, sp.sin):
        for atom in H.atoms(fn):
            arg = atom.args[0]
            for x in (arg.free_symbols & coords):
                if sp.diff(arg, x).free_symbols & coords:
                    bad |= (arg.free_symbols & coords)

    rows = []
    for m in modes:
        warn = None
        if {m.flux, m.charge} & bad:
            warn = ("appears inside a nonlinear cosine (gyrator x nonlinear -> "
                    "coordinate-dependent bracket); canonical quantization is "
                    "ambiguous and the diagonalizer will likely error")
        elif m.kind == FREE:
            warn = "free mode: no confining quadratic or cosine term"
        rows.append((m.flux, m.charge, m.kind, warn))
    return rows


def _is_bias_param(name):
    return name.startswith(("phi_ext", "phi_e", "n_g", "ng_", "q_g", "n_ext",
                            "q_ext", "offset"))


def default_params(result, physical=False):
    """Ordered ``{name: value}`` of default values for every parameter the
    reduced Hamiltonian needs -- its free symbols that are not dynamical
    coordinates.

    Element energies/values get sensible defaults (Josephson/phase-slip energies
    15, capacitances 1, inductances 1, gyrator ratio 0.5); external biases (flux
    ``phi_ext_*`` and offset/gate charges) default to 0.  With ``physical=True``
    the defaults carry units (``15GHz`` / ``70fF`` / ``150nH``) for the GUI's
    physical-units mode.  Element parameters are listed before bias parameters.
    """
    coords = set(result.coordinates)
    params = [s for s in result.H.free_symbols if s not in coords]

    def value(name):
        if _is_bias_param(name):
            return 0
        if name.startswith(("E_J", "E_S", "EJ", "ES")):
            return "15GHz" if physical else 15
        if name.startswith("G"):
            return 0.5
        if name.startswith("L") and not name.startswith("E_L"):
            return "150nH" if physical else 1
        if name.startswith("C") and not name.startswith("E_C"):
            return "70fF" if physical else 1
        return 1                                # E_C / E_L / anything else

    ordered = sorted(params, key=lambda s: (1 if _is_bias_param(str(s)) else 0, str(s)))
    out = {}
    for s in ordered:
        out[str(s)] = value(str(s))
    return out


def _fmt_param_value(v):
    if isinstance(v, str):
        return v
    return str(int(v)) if float(v) == int(v) else str(v)


def param_entry_text(result, existing="", physical=False):
    """Default parameter string for the diagonalization box, preserving any
    values the user already typed for parameters that are still relevant and
    dropping ones that no longer apply."""
    have = {}
    for piece in (existing or "").replace(",", " ").split():
        if "=" in piece:
            k, v = piece.split("=", 1)
            have[k.strip()] = v.strip()
    parts = []
    for name, val in default_params(result, physical=physical).items():
        parts.append(f"{name}={have[name] if name in have else _fmt_param_value(val)}")
    return ", ".join(parts)


_GREEK = {"phi": r"\phi", "varphi": r"\varphi", "theta": r"\theta",
          "psi": r"\psi", "Phi": r"\Phi"}


def _operator_latex(sym):
    """LaTeX for a dynamical variable as a quantum operator: a hat over the
    base symbol with the subscript outside the hat, e.g. ``phi_v2`` ->
    ``\\hat{\\phi}_{v2}`` and ``q_f3`` -> ``\\hat{q}_{f3}``."""
    name = getattr(sym, "name", str(sym))
    base, _, sub = name.partition("_")
    base_tex = _GREEK.get(base, base)
    out = r"\hat{" + base_tex + "}"
    if sub:
        out += "_{" + sub + "}"
    return out


_DBG_PATH = os.path.expanduser("~/fluxcharge_gui.log")


def _dbg(msg):
    """Append a timestamped diagnostic line to ~/fluxcharge_gui.log.

    Off by default; set ``FLUXCHARGE_DEBUG=1`` to enable.  stderr is often
    invisible when the GUI is launched from a console script or bundle, so when
    enabled we log to a file to debug the live app.  Never raises."""
    if not os.environ.get("FLUXCHARGE_DEBUG"):
        return
    try:
        import time
        with open(_DBG_PATH, "a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:
        pass


def _hamiltonian_latex(expr, operators):
    """LaTeX of *expr* with each operator symbol hatted (parameters left as
    ordinary c-numbers)."""
    names = {s: _operator_latex(s) for s in operators}
    return sp.latex(expr, symbol_names=names)


def _split_latex_terms(latex):
    """Split a LaTeX sum into its top-level (brace-depth-0) additive terms,
    each carrying its leading sign, so a long Hamiltonian can be wrapped."""
    terms, cur, depth = [], "", 0
    for ch in latex:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth == 0 and ch in "+-" and cur.strip():
            terms.append(cur.strip())
            cur = ch
        else:
            cur += ch
    if cur.strip():
        terms.append(cur.strip())
    return terms


def _mt_width(latex, fontsize, dpi):
    """Rendered width (px) of a mathtext string, for line wrapping."""
    try:
        from matplotlib.mathtext import MathTextParser
        from matplotlib.font_manager import FontProperties
        return MathTextParser("agg").parse(
            f"${latex}$", dpi=dpi, prop=FontProperties(size=fontsize))[2]
    except Exception:
        return len(latex) * fontsize * 0.55          # crude fallback


def wrap_latex_sum(body, prefix, max_px, fontsize, dpi):
    """Wrap a LaTeX sum *body* (prefixed by *prefix* on the first line) into a
    list of lines each no wider than *max_px*; continuation lines are indented."""
    terms = _split_latex_terms(body) or [body]
    lines, cur = [], (prefix + " " + terms[0]).strip()
    for t in terms[1:]:
        trial = cur + " " + t
        if _mt_width(trial, fontsize, dpi) > max_px:
            lines.append(cur)
            cur = r"\quad " + t
        else:
            cur = trial
    lines.append(cur)
    return lines


# ----------------------------------------------------------------------
# quality-of-life helpers (pure functions -- testable without a display)
# ----------------------------------------------------------------------
def hamiltonian_clipboard(out, fmt="latex", energy=False):
    """Text to copy for the current Hamiltonian / commutators.

    *fmt* is ``"latex"`` (``\\hat{H} = ...``), ``"sympy"`` (a SymPy expression
    string) or ``"commutators"`` (the bracket relations as LaTeX).  *energy*
    selects the familiar-units (E_C/E_L/n) presentation.
    """
    if fmt == "latex":
        return r"\hat{H} = " + (out["H_latex_energy"] if energy else out["H_latex"])
    if fmt == "sympy":
        return str(out["H_energy"] if energy else out["H"])
    if fmt == "commutators":
        return out["commutators_latex_energy"] if energy else out["commutators_latex"]
    raise ValueError(f"unknown clipboard format {fmt!r}")


def eigenenergies_csv(eigenvalues, modes=None):
    """A CSV string of the eigenenergies (and the mode types, as a comment)."""
    lines = []
    if modes:
        lines.append("# modes: " + "; ".join(f"{f}/{c}:{k}" for f, c, k in modes))
    lines.append("level,energy")
    for i, e in enumerate(eigenvalues):
        lines.append(f"{i},{float(e):.12g}")
    return "\n".join(lines) + "\n"


def netlist_error_line(message):
    """Extract the 1-based line number from a NetlistError message, or None."""
    m = re.search(r"line (\d+)", str(message))
    return int(m.group(1)) if m else None


def _session_path():
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "fluxcharge", "session.json")


def load_session():
    """Return the saved session dict (netlist, params, levels, geometry), or {}."""
    try:
        with open(_session_path()) as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_session(data):
    """Persist the session dict; returns True on success.  Never raises."""
    try:
        path = _session_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        return True
    except Exception:
        return False


def _energy_symbol(prefix, value_symbol):
    """Energy-parameter symbol for a capacitance/inductance value symbol:
    ``E_C`` from ``C``, ``E_C1`` from ``C1``, ``E_L`` from ``L`` (a leading
    ``C``/``L`` of the value symbol is folded into the prefix)."""
    name = getattr(value_symbol, "name", str(value_symbol))
    suffix = name[1:] if name[:1] in ("C", "L") else "_" + name
    return sp.Symbol(prefix + suffix, positive=True)


def energy_units_form(H, commutators, capacitances, inductances):
    """Rewrite *H* and *commutators* in the familiar qubit units.

    In this package's natural units (``hbar = 1``, reduced flux quantum ``= 1``,
    hence ``2e = 1``) the charge ``q`` conjugate to the phase ``phi`` is the
    Cooper-pair number ``n`` (``[phi, q] = i`` is ``[phi, n] = i``), so ``q -> n``
    is a relabelling.  Each capacitance ``C`` is written via the charging energy
    ``E_C = e**2/(2C) = 1/(8C)`` (so ``q**2/(2C) -> 4 E_C n**2``) and each
    inductance ``L`` via ``E_L = 1/L`` (so ``phi**2/(2L) -> E_L phi**2/2``).
    ``E_J`` / ``E_S`` are already energies and are left untouched.

    Returns ``(H_energy, commutators_energy, definitions, charge_map)`` where
    *definitions* is a list of ``sympy.Eq`` (``E_C = 1/(8C)``, ``E_L = 1/L``) and
    *charge_map* maps each ``q_*`` symbol to its ``n_*`` relabel.
    """
    subs = {}
    definitions = []
    for C in capacitances:
        EC = _energy_symbol("E_C", C)
        subs[C] = 1 / (8 * EC)
        definitions.append(sp.Eq(EC, sp.Rational(1, 8) / C, evaluate=False))
    for L in inductances:
        EL = _energy_symbol("E_L", L)
        subs[L] = 1 / EL
        definitions.append(sp.Eq(EL, 1 / L, evaluate=False))

    charge_map = {}
    syms = set(H.free_symbols)
    for a, b, _ in commutators:
        syms |= {a, b}
    for s in syms:
        nm = getattr(s, "name", "")
        if nm.startswith("q_"):
            charge_map[s] = sp.Symbol("n_" + nm[2:])

    H_energy = sp.expand(H.subs(subs)).subs(charge_map)
    commutators_energy = [(charge_map.get(a, a), charge_map.get(b, b), v)
                          for a, b, v in commutators]
    return H_energy, commutators_energy, definitions, charge_map


def main():  # pragma: no cover - interactive
    import sys as _sys
    import tkinter as tk
    from tkinter import ttk, filedialog
    import tkinter.font as tkfont

    _MOD = "⌘" if _sys.platform == "darwin" else "Ctrl+"   # ⌘ on macOS

    # Use a stable per-user matplotlib cache dir *before* importing matplotlib so
    # the font cache is built once and reused.  PyInstaller's runtime hook forces
    # MPLCONFIGDIR to a fresh temp dir every launch (and deletes it at exit),
    # which rebuilds the font cache each time (a multi-second delay); override it
    # when frozen.  For a normal run, respect any user-set MPLCONFIGDIR.
    _mpldir = os.path.expanduser("~/.cache/fluxcharge/matplotlib")
    if getattr(_sys, "frozen", False) or not os.environ.get("MPLCONFIGDIR"):
        try:
            os.makedirs(_mpldir, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = _mpldir
        except Exception:
            pass
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.image as mpimg

    # palette
    BG, SURFACE, INK, MUTED = "#eceff4", "#ffffff", "#222a35", "#6b7785"
    ACCENT, ACCENT_HOVER, BORDER, FIELD = "#0e7490", "#0b5566", "#cdd5df", "#ffffff"
    ELEMENT_TYPES = ["C", "L", "J", "QPS"]

    # crisp Computer-Modern-style math
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["figure.dpi"] = 110

    root = tk.Tk()
    root.title("fluxcharge — circuit to Hamiltonian")
    root.geometry("1240x900")
    root.minsize(1040, 720)
    root.configure(bg=BG)

    fams = set(tkfont.families())
    UI = next((f for f in ["Segoe UI", "SF Pro Text", "Helvetica Neue",
                           "Helvetica", "Arial", "DejaVu Sans"] if f in fams), "TkDefaultFont")
    MONO = next((f for f in ["Cascadia Code", "Consolas", "Menlo",
                             "DejaVu Sans Mono", "Courier New"] if f in fams), "TkFixedFont")

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure(".", background=BG, foreground=INK, font=(UI, 10),
                    bordercolor=BORDER, focuscolor=ACCENT)
    style.configure("TFrame", background=BG)
    style.configure("Header.TFrame", background=SURFACE)
    style.configure("TLabel", background=BG, foreground=INK)
    style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=(UI, 9))
    style.configure("Title.TLabel", background=SURFACE, foreground=ACCENT, font=(UI, 18, "bold"))
    style.configure("Subtitle.TLabel", background=SURFACE, foreground=MUTED, font=(UI, 10))
    style.configure("TLabelframe", background=BG, bordercolor=BORDER,
                    relief="solid", borderwidth=1, padding=8)
    style.configure("TLabelframe.Label", background=BG, foreground=MUTED, font=(UI, 9, "bold"))
    style.configure("TButton", background="#dde3ea", foreground=INK,
                    padding=(9, 5), relief="flat", borderwidth=0)
    style.map("TButton", background=[("active", "#cbd4df")])
    style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff",
                    padding=(10, 9), font=(UI, 12, "bold"), relief="flat", borderwidth=0)
    style.map("Accent.TButton", background=[("active", ACCENT_HOVER)])
    style.configure("TEntry", fieldbackground=FIELD, bordercolor=BORDER,
                    insertcolor=INK, padding=4, relief="flat")
    style.configure("TMenubutton", background="#dde3ea", foreground=INK,
                    padding=(8, 4), relief="flat")
    style.map("TMenubutton", background=[("active", "#cbd4df")])
    style.configure("TCheckbutton", background=BG, foreground=INK)
    style.map("TCheckbutton", background=[("active", BG)])
    style.configure("Status.TLabel", background="#e2e6ec", foreground=MUTED, padding=(10, 5))

    def card(parent, title):
        return ttk.LabelFrame(parent, text=title)

    def style_text(widget):
        widget.configure(bg=FIELD, fg=INK, insertbackground=INK, relief="flat",
                         highlightthickness=1, highlightbackground=BORDER,
                         highlightcolor=ACCENT, padx=8, pady=6, selectbackground="#bfe3ee")

    # ---- header ----
    header = ttk.Frame(root, style="Header.TFrame", padding=(16, 12))
    header.pack(fill="x", side="top")
    ttk.Label(header, text="fluxcharge", style="Title.TLabel").pack(anchor="w")
    ttk.Label(header, text="build a circuit  \u2192  schematic, Lagrangian, Hamiltonian, commutators",
              style="Subtitle.TLabel").pack(anchor="w")
    tk.Frame(root, height=1, bg=BORDER).pack(fill="x", side="top")

    # ---- status (bottom): label on the left, progress spinner on the right ----
    style.configure("Status.Horizontal.TProgressbar", background=ACCENT,
                    troughcolor="#e2e6ec", bordercolor="#e2e6ec")
    status_bar = ttk.Frame(root, style="TFrame")
    status_bar.pack(fill="x", side="bottom")
    status = ttk.Label(status_bar, text="ready", style="Status.TLabel", anchor="w")
    status.pack(side="left", fill="x", expand=True)
    progress = ttk.Progressbar(status_bar, mode="indeterminate", length=140,
                               style="Status.Horizontal.TProgressbar")
    # shown (packed) only while a background job runs

    # ---- body ----
    body = ttk.Frame(root, padding=12)
    body.pack(fill="both", expand=True)

    # The left column's stacked cards can be taller than the window (especially on
    # laptop screens), so make it scrollable -- otherwise pack silently clips the
    # bottom cards (the sweep / Sweep / Live controls) off the screen edge.  All
    # the cards still pack into `left`; `left` is now the canvas's inner frame.
    left_outer = ttk.Frame(body, width=486)
    left_outer.pack(side="left", fill="y")
    left_outer.pack_propagate(False)
    left_canvas = tk.Canvas(left_outer, width=470, highlightthickness=0,
                            bg=SURFACE, takefocus=0)
    left_vsb = ttk.Scrollbar(left_outer, orient="vertical", command=left_canvas.yview)
    left_canvas.configure(yscrollcommand=left_vsb.set)
    left_vsb.pack(side="right", fill="y")
    left_canvas.pack(side="left", fill="both", expand=True)
    left = ttk.Frame(left_canvas)
    _left_win = left_canvas.create_window((0, 0), window=left, anchor="nw")
    left.bind("<Configure>",
              lambda _e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
    left_canvas.bind("<Configure>",
                     lambda e: left_canvas.itemconfigure(_left_win, width=e.width))

    def _left_wheel(event):
        # only scroll the left column when the pointer is over it (so the report
        # box on the right keeps its own wheel scrolling)
        w = event.widget
        if str(w).startswith(str(left_outer)):
            step = -1 if event.delta > 0 else 1
            left_canvas.yview_scroll(step, "units")
    root.bind_all("<MouseWheel>", _left_wheel)
    # X11 sends wheel as Button-4/5
    root.bind_all("<Button-4>", lambda e: _left_wheel(type("E", (), {"widget": e.widget, "delta": 120})))
    root.bind_all("<Button-5>", lambda e: _left_wheel(type("E", (), {"widget": e.widget, "delta": -120})))

    right = ttk.Frame(body)
    right.pack(side="left", fill="both", expand=True, padx=(12, 0))

    # ---- netlist card ----
    nl_card = card(left, "Netlist")
    nl_card.pack(fill="x")
    netlist = tk.Text(nl_card, height=10, font=(MONO, 10), wrap="none")
    style_text(netlist)
    netlist.pack(fill="both", expand=True)
    netlist.insert("1.0", EXAMPLE)

    def append_line(line):
        text = netlist.get("1.0", "end-1c")
        if text and not text.endswith("\n"):
            netlist.insert("end", "\n")
        netlist.insert("end", line + "\n")

    fb = ttk.Frame(nl_card); fb.pack(fill="x", pady=(8, 0))

    def do_load():
        path = filedialog.askopenfilename(filetypes=[("netlist", "*.txt *.net *.circuit"), ("all", "*.*")])
        if path:
            with open(path) as fh:
                netlist.delete("1.0", "end"); netlist.insert("1.0", fh.read())

    def do_save():
        path = filedialog.asksaveasfilename(defaultextension=".txt")
        if path:
            with open(path, "w") as fh:
                fh.write(netlist.get("1.0", "end-1c"))

    def do_example():
        netlist.delete("1.0", "end"); netlist.insert("1.0", EXAMPLE)

    def do_clear():
        netlist.delete("1.0", "end")

    for txt, cmd, acc in [("Load", do_load, f"{_MOD}O"), ("Save", do_save, f"{_MOD}S"),
                          ("Example", do_example, None), ("Clear", do_clear, None)]:
        label = f"{txt} ({acc})" if acc else txt
        ttk.Button(fb, text=label, command=cmd).pack(side="left", padx=(0, 4))
    use_latex = tk.BooleanVar(value=False)
    ttk.Checkbutton(fb, text="LaTeX", variable=use_latex,
                    command=lambda: _rerender()).pack(side="right")
    energy_units = tk.BooleanVar(value=False)
    ttk.Checkbutton(fb, text="E_C, E_L, n̂", variable=energy_units,
                    command=lambda: _rerender()).pack(side="right", padx=(0, 8))

    # ---- builder card ----
    b = card(left, "Add to circuit")
    b.pack(fill="x", pady=(10, 0))
    b.columnconfigure(1, weight=1)

    ttk.Label(b, text="element:  type · name · node1 · node2 · value",
              style="Muted.TLabel").grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 3))
    etype = tk.StringVar(value="C")
    ttk.OptionMenu(b, etype, "C", *ELEMENT_TYPES).grid(row=1, column=0, padx=(0, 4), sticky="ew")
    e_name = ttk.Entry(b, width=6); e_name.grid(row=1, column=1, padx=2, sticky="ew"); e_name.insert(0, "e1")
    e_n1 = ttk.Entry(b, width=6); e_n1.grid(row=1, column=2, padx=2); e_n1.insert(0, "v1")
    e_n2 = ttk.Entry(b, width=6); e_n2.grid(row=1, column=3, padx=2); e_n2.insert(0, "v2")
    e_val = ttk.Entry(b, width=7); e_val.grid(row=1, column=4, padx=2); e_val.insert(0, "C")

    def add_element():
        append_line(f"{etype.get()}  {e_name.get()}  {e_n1.get()} {e_n2.get()}  {e_val.get()}".rstrip())
    ttk.Button(b, text="add", command=add_element).grid(row=1, column=5, padx=(4, 0), sticky="ew")

    ttk.Label(b, text="gyrator:  edge1 a b · edge2 a b · ratio",
              style="Muted.TLabel").grid(row=2, column=0, columnspan=6, sticky="w", pady=(10, 3))
    gwrap = ttk.Frame(b); gwrap.grid(row=3, column=0, columnspan=5, sticky="w")
    g_fields = []
    for default in ["e4", "v1", "v3", "e5", "v2", "v3", "G"]:
        en = ttk.Entry(gwrap, width=4); en.pack(side="left", padx=1); en.insert(0, default)
        g_fields.append(en)

    def add_gyrator():
        v = [f.get() for f in g_fields]
        append_line(f"gyrator  {v[0]} {v[1]} {v[2]}   {v[3]} {v[4]} {v[5]}   {v[6]}")
    ttk.Button(b, text="add", command=add_gyrator).grid(row=3, column=5, padx=(4, 0), sticky="ew")

    ttk.Label(b, text="loop:  name · signed edges",
              style="Muted.TLabel").grid(row=4, column=0, columnspan=6, sticky="w", pady=(10, 3))
    l_name = ttk.Entry(b, width=6); l_name.grid(row=5, column=0, padx=(0, 2)); l_name.insert(0, "f1")
    l_edges = ttk.Entry(b); l_edges.grid(row=5, column=1, columnspan=4, padx=2, sticky="ew"); l_edges.insert(0, "+e1 -e2")

    def add_loop():
        append_line(f"loop  {l_name.get()}  {l_edges.get()}")
    ttk.Button(b, text="add", command=add_loop).grid(row=5, column=5, padx=(4, 0), sticky="ew")

    ttk.Label(b, text="gauge:  ground node · open loop",
              style="Muted.TLabel").grid(row=6, column=0, columnspan=6, sticky="w", pady=(10, 3))
    g_ground = ttk.Entry(b, width=6); g_ground.grid(row=7, column=0, padx=(0, 2)); g_ground.insert(0, "v1")
    g_open = ttk.Entry(b, width=6); g_open.grid(row=7, column=1, padx=2, sticky="w")

    def add_gauge():
        if g_ground.get().strip():
            append_line(f"ground {g_ground.get().strip()}")
        if g_open.get().strip():
            append_line(f"open   {g_open.get().strip()}")
    ttk.Button(b, text="set", command=add_gauge).grid(row=7, column=5, padx=(4, 0), sticky="ew")

    gen_btn = ttk.Button(left, text=f"Generate  \u2192   ({_MOD}\u21a9 / F5)",
                         style="Accent.TButton", command=lambda: generate())
    gen_btn.pack(fill="x", pady=(12, 0))
    dual_btn = ttk.Button(left, text=f"Dualize  \u21c4   ({_MOD}D)",
                          command=lambda: dualize())
    dual_btn.pack(fill="x", pady=(6, 0))

    # ---- numerics card ----
    num_card = card(left, "Numerical diagonalization")
    num_card.pack(fill="x", pady=(10, 0))
    num_card.columnconfigure(1, weight=1)
    ttk.Label(num_card, text="params:  E_J=15, C=1",
              style="Muted.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
    params_entry = ttk.Entry(num_card)
    params_entry.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 4))
    params_entry.insert(0, "E_J=15, C=1, G=0.5")
    ttk.Label(num_card, text="levels:", style="Muted.TLabel").grid(row=2, column=0, sticky="w", pady=(4, 0))
    levels_entry = ttk.Entry(num_card, width=5)
    levels_entry.grid(row=2, column=1, sticky="w", pady=(4, 0))
    levels_entry.insert(0, "6")
    diag_btn = ttk.Button(num_card, text=f"Diagonalize  ({_MOD}K)",
                          command=lambda: diagonalize())
    diag_btn.grid(row=1, column=2, rowspan=2, sticky="ns", padx=(4, 0))

    units_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(num_card, text="physical units (fF / nH / GHz → spectrum in GHz)",
                    variable=units_var).grid(row=3, column=0, columnspan=3,
                                             sticky="w", pady=(4, 0))

    ttk.Label(num_card, text="cutoffs:", style="Muted.TLabel").grid(
        row=4, column=0, sticky="w", pady=(4, 0))
    cutoffs_entry = ttk.Entry(num_card)
    cutoffs_entry.grid(row=4, column=1, columnspan=2, sticky="ew", pady=(4, 0))
    cutoffs_entry.insert(0, "")     # e.g. "phi_v2=120, q_f1=61"; blank = defaults
    ttk.Label(num_card, text="wavefunctions:", style="Muted.TLabel").grid(
        row=5, column=0, sticky="w", pady=(4, 0))
    wf_rep = tk.StringVar(value="auto")
    ttk.OptionMenu(num_card, wf_rep, "auto", "auto", "flux", "charge").grid(
        row=5, column=1, sticky="w", pady=(4, 0))

    ttk.Label(num_card, text="sweep:  parameter · from · to · points",
              style="Muted.TLabel").grid(row=6, column=0, columnspan=3,
                                         sticky="w", pady=(8, 2))
    sweep_wrap = ttk.Frame(num_card)
    sweep_wrap.grid(row=7, column=0, columnspan=3, sticky="ew")
    sweep_param = ttk.Entry(sweep_wrap, width=9); sweep_param.pack(side="left", padx=1)
    sweep_param.insert(0, "E_J")
    sweep_from = ttk.Entry(sweep_wrap, width=5); sweep_from.pack(side="left", padx=1)
    sweep_from.insert(0, "1")
    sweep_to = ttk.Entry(sweep_wrap, width=5); sweep_to.pack(side="left", padx=1)
    sweep_to.insert(0, "30")
    sweep_pts = ttk.Entry(sweep_wrap, width=5); sweep_pts.pack(side="left", padx=1)
    sweep_pts.insert(0, "41")
    sweep_quantity = tk.StringVar(value="levels")
    ttk.OptionMenu(sweep_wrap, sweep_quantity, "levels",
                   "levels", "transitions", "anharmonicity").pack(side="left", padx=2)
    # buttons on their own row so neither is clipped in the narrow left panel
    sweep_btns = ttk.Frame(num_card)
    sweep_btns.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(4, 0))
    ttk.Button(sweep_btns, text="Sweep (static plot)",
               command=lambda: sweep_plot()).pack(side="left", padx=(0, 4))
    ttk.Button(sweep_btns, text="Live (interactive sliders)",
               command=lambda: live_explore()).pack(side="left")

    # ---- right: outputs ----
    fig = Figure(figsize=(6.8, 5.4))
    fig.patch.set_facecolor(SURFACE)
    ax_sch = fig.add_axes([0.03, 0.50, 0.94, 0.45]); ax_sch.axis("off")
    ax_h = fig.add_axes([0.03, 0.27, 0.94, 0.20]); ax_h.axis("off")
    ax_comm = fig.add_axes([0.03, 0.03, 0.94, 0.21]); ax_comm.axis("off")
    ax_sch.text(0.5, 0.5, "press Generate", ha="center", va="center", color="#aab2bd")
    right.rowconfigure(0, weight=1)
    right.rowconfigure(1, weight=0, minsize=150)
    right.columnconfigure(0, weight=1)

    canvas_border = tk.Frame(right, bg=BORDER)
    canvas_border.grid(row=0, column=0, sticky="nsew")
    canvas = FigureCanvasTkAgg(fig, master=canvas_border)
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=1, pady=1)

    rep_card = card(right, "details")
    rep_card.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
    rep_vsb = ttk.Scrollbar(rep_card, orient="vertical")
    rep_vsb.pack(side="right", fill="y")
    report = tk.Text(rep_card, height=8, font=(MONO, 9), wrap="none", yscrollcommand=rep_vsb.set)
    style_text(report)
    report.pack(side="left", fill="both", expand=True)
    rep_vsb.config(command=report.yview)

    last = {"out": None, "text": None}
    last_diag = {"summary": None}      # most recent diagonalization, for CSV export
    busy_state = {"n": 0}
    action_buttons = [gen_btn, dual_btn, diag_btn]

    def busy_on(text):
        # NOTE: we deliberately do NOT disable the action buttons here.  On macOS
        # (Aqua) a ttk button left in the "disabled" state has been observed not
        # to recover its clickability after "!disabled", which made Generate stop
        # responding.  The spinner + cursor + status are the activity indicator.
        busy_state["n"] += 1
        root.configure(cursor="watch")
        if not progress.winfo_ismapped():
            progress.pack(side="right", padx=10, pady=4)
            progress.start(12)
        status.config(text=text, foreground=MUTED)

    def busy_off():
        busy_state["n"] = max(0, busy_state["n"] - 1)
        if busy_state["n"] == 0:
            progress.stop()
            progress.pack_forget()
            root.configure(cursor="")

    def clear_error():
        netlist.tag_remove("errline", "1.0", "end")

    def report_error(msg):
        """Surface an error inline (no modal): red status, details panel, and a
        highlight of the offending netlist line if the message carries one."""
        msg = str(msg)
        status.config(text=f"error: {msg}", foreground="#b00020")
        report.delete("1.0", "end")
        report.insert("end", "ERROR\n\n" + msg)
        ln = netlist_error_line(msg)
        if ln:
            netlist.tag_remove("errline", "1.0", "end")
            netlist.tag_config("errline", background="#fde0e0")
            netlist.tag_add("errline", f"{ln}.0", f"{ln}.end+1c")
            netlist.see(f"{ln}.0")

    def run_async(work, on_success, busy_text="computing…"):
        """Run *work()* then *on_success(result)* on the main thread, off the
        click handler.

        Tkinter on macOS (Aqua) must be serviced on the main thread, and running
        a background worker thread alongside live UI events can wedge the event
        loop so clicks stop registering.  So instead of a thread we show the busy
        state, return from the click handler (which lets Tk paint it), and do the
        work in a follow-up ``after`` callback.  The window is briefly busy while
        the work runs, but the buttons always respond."""
        busy_on(busy_text)
        _dbg(f"run_async: busy_on ({busy_text}); scheduling work")

        def do():
            import traceback
            _dbg("run_async.do: work() start")
            try:
                result = work()
            except Exception as exc:
                _dbg("run_async.do: work() RAISED\n" + traceback.format_exc())
                busy_off()
                report_error(exc)
                return
            _dbg("run_async.do: work() done; busy_off; on_success")
            busy_off()
            try:
                on_success(result)
                _dbg("run_async.do: on_success done")
            except Exception as exc:
                _dbg("run_async.do: on_success RAISED\n" + traceback.format_exc())
                report_error(exc)

        # let the busy state paint (the 30 ms tick is enough), then run on the
        # main thread.  No update_idletasks() here -- calling it before mainloop
        # has started can wedge event handling on macOS.
        root.after(30, do)

    def _draw_panels(out):
        ax_sch.clear(); ax_sch.axis("off")
        if out["schematic"] and os.path.exists(out["schematic"]):
            ax_sch.imshow(mpimg.imread(out["schematic"]))
        ax_sch.set_title((out["title"] or "circuit").replace("_", r"\_"),
                         fontsize=12, color=INK, fontfamily="sans-serif")

        e_units = bool(energy_units.get())
        H_l = out["H_latex_energy"] if e_units else out["H_latex"]
        comm_l = out["commutators_latex_energy"] if e_units else out["commutators_latex"]

        ax_h.clear(); ax_h.axis("off")
        label = "Hamiltonian (energy units)" if e_units else "Hamiltonian"
        ax_h.text(0.0, 0.98, label, transform=ax_h.transAxes, ha="left",
                  va="top", fontsize=9, color=MUTED, fontfamily="sans-serif")
        # wrap a long Hamiltonian over several lines and shrink to fit the panel
        try:
            dpi = fig.dpi
            pos = ax_h.get_position()
            fw, fh = fig.get_size_inches()
            max_px = pos.width * fw * dpi * 0.97
            avail_h = pos.height * fh * dpi * 0.74      # leave room for the label
            lines, fs = None, 16
            for fs in range(16, 5, -1):
                lines = wrap_latex_sum(H_l, r"\hat{H} =", max_px, fs, dpi)
                if len(lines) * fs * 1.7 * dpi / 72.0 <= avail_h:
                    break
            n = len(lines)
            ys = [0.55] if n == 1 else [0.78 - i * (0.72 / (n - 1)) for i in range(n)]
            for line, y in zip(lines, ys):
                ax_h.text(0.5, y, f"${line}$", ha="center", va="center",
                          fontsize=fs, color=INK)
        except Exception:
            ax_h.text(0.5, 0.55, f"$\\hat{{H}} = {H_l}$", ha="center",
                      va="center", fontsize=12, color=INK)
        if e_units and out["energy_defs_latex"]:
            ax_h.text(0.5, 0.03, f"$({out['energy_defs_latex']})$",
                      ha="center", va="center", fontsize=8, color=MUTED)

        ax_comm.clear(); ax_comm.axis("off")
        if comm_l:
            ax_comm.text(0.0, 0.95, "commutation relations", transform=ax_comm.transAxes,
                         ha="left", va="top", fontsize=9, color=MUTED, fontfamily="sans-serif")
            ax_comm.text(0.5, 0.50, f"${comm_l}$",
                         ha="center", va="center", fontsize=15, color=INK)
            if out["compact"]:
                names = ", ".join(f"${_operator_latex(s)}$" for s in out["compact"])
                ax_comm.text(0.5, 0.08,
                             f"({names} sit inside a cosine; may live on $S^1$, "
                             "then use the exponential form)",
                             ha="center", va="center", fontsize=8, color=MUTED,
                             fontfamily="sans-serif")
        canvas.draw()
        # force the Tk canvas to blit now (idle tasks only, so no event
        # reentrancy); without this the redraw can lag on macOS and a click
        # looks like it did nothing
        root.update_idletasks()

    def _rerender():
        out = last["out"]
        if out is None:
            return
        want = bool(use_latex.get())
        matplotlib.rcParams["text.usetex"] = want
        try:
            _draw_panels(out)
            status.config(text="rendered with system LaTeX" if want else "generated",
                          foreground="#0a7d2c")
        except Exception as exc:
            if want:   # no TeX install / render failure: fall back to mathtext
                matplotlib.rcParams["text.usetex"] = False
                use_latex.set(False)
                try:
                    _draw_panels(out)
                except Exception:
                    pass
                status.config(text=f"system LaTeX unavailable; using mathtext", foreground="#b26a00")
            else:
                status.config(text=f"render error: {exc}", foreground="#b00020")

    def generate():
        _dbg("generate() clicked")
        text = netlist.get("1.0", "end-1c")
        if last["out"] is not None and last["text"] == text:
            # circuit unchanged since the last Generate -- nothing to recompute;
            # makes repeated clicks instant instead of piling up redundant work
            import time
            status.config(
                text=f"✓ up to date: {last['out']['title'] or 'circuit'}"
                     f"   ({time.strftime('%H:%M:%S')})", foreground="#0a7d2c")
            _dbg("generate(): netlist unchanged; skipped recompute")
            return

        def work():
            # pure-sympy reduction off the main thread; no matplotlib here
            return compute(text, canonical=True, draw=False)

        def done(out):
            # the schematic uses schemdraw/matplotlib, so it must be drawn on the
            # main thread; show feedback because this part can briefly block
            clear_error()
            status.config(text="rendering…", foreground=MUTED)
            root.update_idletasks()
            try:
                fd, p = tempfile.mkstemp(suffix=".png", prefix="fluxcharge_")
                os.close(fd)
                out["circuit"].schematic(path=p)
                out["schematic"] = p
            except Exception:
                out["schematic"] = None
            finally:
                # schemdraw draws via pyplot; close its stray figures so they
                # don't accumulate over repeated Generates
                try:
                    import matplotlib.pyplot as plt
                    plt.close("all")
                except Exception:
                    pass
            last["out"] = out
            last["text"] = text
            # pre-fill the diagonalization box with default values for every
            # parameter this circuit needs (keeping any values already typed)
            try:
                filled = param_entry_text(out["result"], existing=params_entry.get(),
                                          physical=units_var.get())
                params_entry.delete(0, "end")
                params_entry.insert(0, filled)
            except Exception:
                pass
            _rerender()
            report.delete("1.0", "end")
            report.insert("end", f"H = {out['H']}\n\nLagrangian:\n"
                          f"{out['lagrangian']}\n\n{out['report']}")
            import time
            title = out["title"] or "circuit"
            root.title(f"fluxcharge — {title}")
            status.config(text=f"✓ generated: {title}"
                          f"   ({time.strftime('%H:%M:%S')})", foreground="#0a7d2c")

        run_async(work, done, busy_text="reducing circuit…")

    def dualize():
        text = netlist.get("1.0", "end-1c")
        try:
            ckt = from_netlist(text)
            ckt.validate()
            d = dual(ckt)
            netlist.delete("1.0", "end")
            netlist.insert("1.0", to_netlist(d))
            clear_error()
        except Exception as exc:
            report_error(exc)
            return
        generate()

    def _parse_physical(raw):
        """``"C=70fF, E_J=15GHz"`` -> ``{"C":"70fF","E_J":"15GHz"}`` (units kept)."""
        out = {}
        for piece in raw.replace(",", " ").split():
            if "=" in piece:
                k, v = piece.split("=", 1)
                out[k.strip()] = v.strip()
        return out

    def _diag_params(text):
        """Numeric params from the box: physical (fF/nH/GHz -> GHz spectrum) when
        the units box is ticked, else raw natural-unit numbers."""
        raw = params_entry.get()
        if units_var.get():
            from .units import to_natural
            return to_natural(from_netlist(text), _parse_physical(raw))
        from .__main__ import _parse_params
        return _parse_params([raw])

    def _cutoffs():
        """Parse the cutoffs box (``"phi_v2=120, q_f1=61"``) into {name: int}, or
        None for default basis sizes."""
        out = {}
        for piece in cutoffs_entry.get().replace(",", " ").split():
            if "=" in piece:
                k, v = piece.split("=", 1)
                out[k.strip()] = int(float(v))
        return out or None

    def _current_result(text):
        """The reduction for *text*, reusing the last Generate when unchanged."""
        if last["text"] == text and last["out"]:
            return last["out"]["result"]
        ckt = from_netlist(text)
        ckt.validate()
        return ckt.hamiltonian(ground=getattr(ckt, "ground", None),
                               open_loops=getattr(ckt, "open_loops", None) or None,
                               canonical=True)

    def _ask_mode_types(result):
        """Modal dialog: per conjugate pair, choose extended vs compact, defaulted
        to the automatic classification, with warnings for the subtle cases.
        Returns the mode_types dict (str(flux) -> kind), or None if cancelled."""
        rows = mode_type_options(result)
        if not rows:
            return {}
        label2val = {lab: val for val, lab in MODE_KIND_LABELS}
        val2label = {val: lab for val, lab in MODE_KIND_LABELS}
        labels = [lab for _v, lab in MODE_KIND_LABELS]

        win = tk.Toplevel(root)
        win.title("Declare mode types")
        win.transient(root)
        win.configure(bg=BG)
        ttk.Label(win, wraplength=470, style="Muted.TLabel",
                  text="Choose how each mode is quantized. Compactness is physical "
                       "input (an island's phase lives on a circle, an inductive "
                       "loop's flux on the line). Defaults are the automatic "
                       "classification — change one only if you know its "
                       "compactness."
                  ).grid(row=0, column=0, columnspan=2, padx=12, pady=(12, 8), sticky="w")
        combos = []
        r = 1
        for flux, charge, default_kind, warn in rows:
            ttk.Label(win, text=f"{flux}  ↔  {charge}").grid(
                row=r, column=0, sticky="w", padx=12, pady=2)
            cb = ttk.Combobox(win, values=labels, state="readonly", width=30)
            cb.set(val2label.get(default_kind, labels[0]))
            cb.grid(row=r, column=1, sticky="w", padx=12, pady=2)
            combos.append((str(flux), cb))
            r += 1
            if warn:
                ttk.Label(win, text="⚠  " + warn, wraplength=450,
                          foreground="#b25c00", background=BG).grid(
                    row=r, column=0, columnspan=2, sticky="w", padx=28, pady=(0, 4))
                r += 1

        holder = {"mt": None, "ok": False}

        def on_ok():
            holder["mt"] = {flux: label2val[cb.get()] for flux, cb in combos}
            holder["ok"] = True
            win.destroy()

        btns = ttk.Frame(win)
        btns.grid(row=r, column=0, columnspan=2, pady=12)
        ttk.Button(btns, text="Diagonalize", command=on_ok).pack(side="left", padx=5)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="left", padx=5)
        win.bind("<Return>", lambda _e: on_ok())
        win.bind("<Escape>", lambda _e: win.destroy())
        win.grab_set()
        root.wait_window(win)
        return holder["mt"] if holder["ok"] else None

    def diagonalize():
        text = netlist.get("1.0", "end-1c")
        try:
            params = _diag_params(text)
            n = max(1, int(levels_entry.get() or 6))
            cutoffs = _cutoffs()
        except Exception as exc:
            report_error(exc)
            return
        unit = "GHz" if units_var.get() else ""

        # reduce first (reuse the last Generate when unchanged) so the dialog can
        # list the modes; this also lets the diagonalization reuse the reduction
        try:
            res = _current_result(text)
        except Exception as exc:
            report_error(exc)
            return

        mode_types = _ask_mode_types(res)
        if mode_types is None:
            return                                  # user cancelled

        def work():
            return summary_from_result(res, params, n_levels=n, cutoffs=cutoffs,
                                       mode_types=mode_types)

        run_async(work, lambda summ: _show_diag(summ, params, n, unit, cutoffs),
                  busy_text="diagonalizing…")

    def _show_diag(summ, params, n, unit="", cutoffs=None):
        import re as _re
        clear_error()
        last_diag["summary"] = summ
        res = summ["result"]
        ev = summ["eigenenergies"]
        u = f" {unit}" if unit else ""
        lines = ["Mode types:"]
        for flux, charge, kind in summ["modes"]:
            lines.append(f"  {flux} / {charge}: {kind}")
        lines.append("\nEigenenergies:")
        lines += [f"  E_{i} = {e:.6g}{u}" for i, e in enumerate(ev)]
        if summ["transitions"]:
            lines.append("transitions above ground: "
                         + ", ".join(f"{t:.5g}{u}" for t in summ["transitions"]))
        # exact charge matrix elements |<0|n|1>| for each mode's charge
        me = []
        for _flux, charge, _kind in summ["modes"]:
            try:
                M = res.matrix_elements(charge, params, n_levels=min(n, 4),
                                        cutoffs=cutoffs)
                me.append(f"  |<0|{charge}|1>| = {abs(M[0, 1]):.4f}")
            except Exception:
                pass
        if me:
            lines.append("\nmatrix elements:")
            lines += me
        # sensitivity of f01 to any external bias (zero => sweet spot)
        biases = [str(k) for k in params if _re.match(r"(phi_ext_|n_g_)", str(k))]
        if biases:
            lines.append("\nbias sensitivity (d f01/d bias):")
            for bsym in biases:
                try:
                    df, _d2 = res.transition_sensitivity(bsym, params, cutoffs=cutoffs)
                    tag = "   <- sweet spot" if abs(df) < 1e-3 else ""
                    lines.append(f"  {bsym}: {df:+.4f}{u}/unit{tag}")
                except Exception:
                    pass
        report.delete("1.0", "end")
        report.insert("end", "\n".join(lines))
        status.config(text=f"diagonalized{(' (' + unit + ')') if unit else ''}",
                      foreground="#0a7d2c")

        # popup figure: potential + wavefunctions (single mode) or a level diagram
        win = tk.Toplevel(root)
        win.title("fluxcharge — spectrum")
        win.configure(bg=SURFACE)
        pfig = Figure(figsize=(6.0, 4.6))
        pfig.patch.set_facecolor(SURFACE)
        pax = pfig.add_subplot(111)
        # prefer the potential + wavefunctions view; it needs H = kinetic + V(phi),
        # which fails for a gyrator's phi*q cross term -- fall back to a level diagram.
        drew = False
        if summ["single_mode"]:
            try:
                summ["result"].plot_potential_wavefunctions(
                    params, n_levels=n, ax=pax, cutoffs=cutoffs,
                    representation=wf_rep.get())
                drew = True
            except NotImplementedError:
                status.config(
                    text="flux–charge coupling (gyrator); showing levels instead",
                    foreground="#b26a00")
        if not drew:
            try:
                summ["result"].plot_energy_levels(params, n_levels=n, ax=pax,
                                                  cutoffs=cutoffs)
            except Exception as exc:
                pax.text(0.5, 0.5, f"plot unavailable:\n{exc}", ha="center",
                         va="center", wrap=True, fontsize=9)
        pcanvas = FigureCanvasTkAgg(pfig, master=win)
        pcanvas.get_tk_widget().pack(fill="both", expand=True, side="top")

        bar = ttk.Frame(win)
        bar.pack(fill="x", side="bottom")

        def save_plot():
            path = filedialog.asksaveasfilename(
                parent=win, defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
            if path:
                pfig.savefig(path, dpi=200, bbox_inches="tight")
                status.config(text=f"saved plot to {os.path.basename(path)}",
                              foreground="#0a7d2c")

        ttk.Button(bar, text="Save plot…", command=save_plot).pack(side="right", padx=6, pady=6)
        ttk.Button(bar, text="Export eigenenergies (CSV)…",
                   command=lambda: export_csv()).pack(side="right", padx=2, pady=6)
        pcanvas.draw()

    def sweep_plot():
        """Plot the spectrum vs the chosen parameter/bias over a range."""
        import numpy as np
        text = netlist.get("1.0", "end-1c")
        try:
            n = max(1, int(levels_entry.get() or 6))
            base = _diag_params(text)
            param = sweep_param.get().strip()
            lo, hi = float(sweep_from.get()), float(sweep_to.get())
            pts = max(2, int(sweep_pts.get() or 41))
            cutoffs = _cutoffs()
            res = _current_result(text)
        except Exception as exc:
            report_error(exc)
            return
        unit = "GHz" if units_var.get() else ""
        quantity = sweep_quantity.get()
        vals = np.linspace(lo, hi, pts)
        busy_on(f"sweeping {param}…")

        def do():
            win = tk.Toplevel(root)
            win.title(f"fluxcharge — sweep {param}")
            win.configure(bg=SURFACE)
            f = Figure(figsize=(6.2, 4.4))
            f.patch.set_facecolor(SURFACE)
            ax = f.add_subplot(111)
            try:
                res.plot_spectrum(param, vals, base, n_levels=n, ax=ax,
                                  quantity=quantity, cutoffs=cutoffs)
                if unit:
                    ax.set_ylabel(ax.get_ylabel() + f"  ({unit})")
            except Exception as exc:
                busy_off()
                win.destroy()
                report_error(exc)
                return
            busy_off()
            c = FigureCanvasTkAgg(f, master=win)
            c.get_tk_widget().pack(fill="both", expand=True, side="top")

            def save_sweep():
                path = filedialog.asksaveasfilename(
                    parent=win, defaultextension=".png",
                    filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
                if path:
                    f.savefig(path, dpi=200, bbox_inches="tight")
            ttk.Button(win, text="Save plot…", command=save_sweep).pack(
                side="bottom", anchor="e", padx=6, pady=6)
            c.draw()
            status.config(text=f"swept {param} ({len(vals)} pts)", foreground="#0a7d2c")

        root.after(30, do)

    def live_explore():
        """Open an interactive spectrum window with live parameter sliders,
        embedded in Tk (so the sliders are responsive inside the app)."""
        from .interactive import (spectrum_vs_param, ranges_from_params,
                                   parameter_symbols)
        text = netlist.get("1.0", "end-1c")
        try:
            n = max(2, int(levels_entry.get() or 6))
            base = _diag_params(text)
            cutoffs = _cutoffs()
            res = _current_result(text)
        except Exception as exc:
            report_error(exc)
            return
        quantity = sweep_quantity.get()
        if quantity not in ("levels", "transitions"):
            quantity = "levels"                     # 'anharmonicity' has no live view
        names = {str(s) for s in parameter_symbols(res)}
        if not names:
            status.config(text="no free parameters to slide", foreground="#b26a00")
            return
        want = sweep_param.get().strip()
        sweep = want if want in names else None     # else auto-pick (a bias, ...)
        ranges = ranges_from_params(res, base)
        busy_on("building live explorer…")

        def do():
            win = tk.Toplevel(root)
            win.title("fluxcharge — live spectrum")
            win.configure(bg=SURFACE)
            efig = Figure(figsize=(7.6, 5.2))
            efig.patch.set_facecolor("white")
            ecanvas = FigureCanvasTkAgg(efig, master=win)
            ecanvas.get_tk_widget().pack(fill="both", expand=True, side="top")
            try:
                # draw into our Tk-bound figure so the sliders are live in-app
                spectrum_vs_param(res, sweep=sweep, ranges=ranges, n_levels=n,
                                  cutoffs=cutoffs, quantity=quantity,
                                  weight_by=True, fig=efig, show=False)
            except Exception as exc:
                busy_off(); win.destroy(); report_error(exc); return
            busy_off()

            def save_live():
                path = filedialog.asksaveasfilename(
                    parent=win, defaultextension=".png",
                    filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
                if path:
                    efig.savefig(path, dpi=200, bbox_inches="tight")
            ttk.Button(win, text="Save plot…", command=save_live).pack(
                side="bottom", anchor="e", padx=6, pady=6)
            ecanvas.draw()
            status.config(text="live explorer — drag a slider", foreground="#0a7d2c")

        root.after(30, do)

    # ---- copy / export actions ----
    def copy_h(fmt):
        if not last["out"]:
            status.config(text="nothing to copy — press Generate first",
                          foreground="#b26a00")
            return
        txt = hamiltonian_clipboard(last["out"], fmt, energy=bool(energy_units.get()))
        root.clipboard_clear()
        root.clipboard_append(txt)
        status.config(text=f"copied {fmt} to clipboard", foreground="#0a7d2c")

    def save_schematic():
        out = last["out"]
        if not out or not out.get("schematic") or not os.path.exists(out["schematic"]):
            status.config(text="generate a circuit first", foreground="#b26a00")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if not path:
            return
        try:
            import shutil
            if path.lower().endswith(".png"):
                shutil.copyfile(out["schematic"], path)   # already a PNG
            else:
                out["circuit"].schematic(path=path)        # re-render at the format
            status.config(text=f"saved schematic to {os.path.basename(path)}",
                          foreground="#0a7d2c")
        except Exception as exc:
            report_error(exc)

    def export_csv():
        summ = last_diag["summary"]
        if not summ:
            status.config(text="diagonalize first to export eigenenergies",
                          foreground="#b26a00")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            with open(path, "w") as fh:
                fh.write(eigenenergies_csv(summ["eigenenergies"], summ["modes"]))
            status.config(text=f"saved {os.path.basename(path)}", foreground="#0a7d2c")
        except Exception as exc:
            report_error(exc)

    def import_scqubits():
        from .interop import from_scqubits_yaml
        from .netlist import to_netlist
        path = filedialog.askopenfilename(
            filetypes=[("scqubits YAML", "*.yaml *.yml"), ("all", "*.*")])
        if not path:
            return
        try:
            imp, params = from_scqubits_yaml(path)
            netlist.delete("1.0", "end")
            netlist.insert("1.0", to_netlist(imp))
            if params:           # prefill the params box so it diagonalizes in GHz
                params_entry.delete(0, "end")
                params_entry.insert(0, ", ".join(f"{k}={v}" for k, v in params.items()))
                units_var.set(False)
            clear_error()
            generate()
            status.config(text=f"imported {os.path.basename(path)} from scqubits",
                          foreground="#0a7d2c")
        except Exception as exc:
            report_error(exc)

    def export_qutip():
        text = netlist.get("1.0", "end-1c")
        try:
            ckt = from_netlist(text)
            params = _diag_params(text)
        except Exception as exc:
            report_error(exc)
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".py", filetypes=[("Python", "*.py")])
        if not path:
            return
        title = getattr(ckt, "title", None) or "circuit"
        ground = getattr(ckt, "ground", None)
        opens = getattr(ckt, "open_loops", None) or None
        script = f'''"""QuTiP model of {title} -- generated by fluxcharge.

Run with qutip + fluxcharge installed.  H and the mode operators are qutip.Qobj,
so the full QuTiP engine (mesolve, sesolve, steadystate, expect, ...) is
available -- including for gyrator / phase-slip circuits.
"""
import numpy as np
import qutip
from fluxcharge import from_netlist

NETLIST = """\\
{text}
"""
PARAMS = {params!r}

ckt = from_netlist(NETLIST)
res = ckt.hamiltonian(ground={ground!r}, open_loops={opens!r},
                      strict=False, canonical=True)
model = res.to_qutip(PARAMS)        # pass cutoffs={{"phi_x": N}} to refine the basis
H = model["H"]                      # Hamiltonian (qutip.Qobj)
ops = model["operators"]            # per-mode charge/flux operators (qutip.Qobj)

print("modes:", model["modes"])
print("lowest levels:", np.round(np.sort(H.eigenenergies())[:6], 4))

# --- example: open-system relaxation (edit to taste) ---
# evals, evecs = H.eigenstates()
# c_ops = [0.02 * next(iter(ops.values()))]            # a collapse operator
# tlist = np.linspace(0, 200, 400)
# out = qutip.mesolve(H, evecs[1], tlist, c_ops=c_ops, e_ops=[evecs[1].proj()])
'''
        try:
            with open(path, "w") as fh:
                fh.write(script)
            status.config(text=f"wrote QuTiP script {os.path.basename(path)}",
                          foreground="#0a7d2c")
        except Exception as exc:
            report_error(exc)

    # Pre-warm matplotlib's mathtext font cache shortly after the window is up.
    # The first equation render builds this cache and can take a few seconds; we
    # warm it on the main thread (no threads -- see run_async) so the first real
    # render is fast.  It runs once, after startup, off a brief ``after`` tick.
    def _prewarm_fonts():
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            f = Figure()
            FigureCanvasAgg(f)
            f.add_axes([0, 0, 1, 1]).text(
                0.5, 0.5, r"$\hat{H}=\frac{\hat{q}^2}{2C}-E_J\cos\hat{\phi}$")
            f.canvas.draw()
        except Exception:
            pass

    # Keyboard shortcuts -- robust even when the OS is flaky about delivering
    # mouse clicks to a terminal-launched (non-bundled) app on macOS.  Bound on
    # both Command (macOS) and Control (Windows/Linux) so the app is portable.
    def _key(fn):
        return lambda event=None: (fn(), "break")[1]
    for seq in ("<Command-Return>", "<Control-Return>", "<F5>",
                "<Command-g>", "<Control-g>"):
        root.bind_all(seq, _key(generate))
    for seq in ("<Command-d>", "<Control-d>"):
        root.bind_all(seq, _key(dualize))
    for seq in ("<Command-k>", "<Control-k>"):
        root.bind_all(seq, _key(diagonalize))
    for seq in ("<Command-o>", "<Control-o>"):
        root.bind_all(seq, _key(do_load))
    for seq in ("<Command-s>", "<Control-s>"):
        root.bind_all(seq, _key(do_save))

    # ---- session persistence ----
    def _collect_session():
        return {
            "netlist": netlist.get("1.0", "end-1c"),
            "params": params_entry.get(),
            "levels": levels_entry.get(),
            "geometry": root.winfo_geometry(),
            "energy_units": bool(energy_units.get()),
        }

    def _restore_session():
        data = load_session()
        if not data:
            return
        if data.get("geometry"):
            try:
                root.geometry(data["geometry"])
            except Exception:
                pass
        if data.get("netlist", "").strip():
            netlist.delete("1.0", "end")
            netlist.insert("1.0", data["netlist"])
        if data.get("params"):
            params_entry.delete(0, "end")
            params_entry.insert(0, data["params"])
        if data.get("levels"):
            levels_entry.delete(0, "end")
            levels_entry.insert(0, str(data["levels"]))
        if data.get("energy_units"):
            energy_units.set(True)

    def on_quit():
        save_session(_collect_session())
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_quit)
    for seq in ("<Command-q>", "<Control-q>", "<Command-w>", "<Control-w>"):
        root.bind_all(seq, _key(on_quit))

    # ---- menu bar (discoverable actions + accelerators) ----
    menubar = tk.Menu(root)
    m_file = tk.Menu(menubar, tearoff=0)
    m_file.add_command(label="Open Netlist…", command=do_load, accelerator=f"{_MOD}O")
    m_file.add_command(label="Save Netlist…", command=do_save, accelerator=f"{_MOD}S")
    m_file.add_command(label="Import scqubits YAML…", command=import_scqubits)
    m_file.add_separator()
    m_file.add_command(label="Save Schematic…", command=save_schematic)
    m_file.add_command(label="Export Eigenenergies (CSV)…", command=export_csv)
    m_file.add_command(label="Export QuTiP script…", command=export_qutip)
    m_file.add_separator()
    m_file.add_command(label="Quit", command=on_quit, accelerator=f"{_MOD}Q")
    menubar.add_cascade(label="File", menu=m_file)

    m_edit = tk.Menu(menubar, tearoff=0)
    m_edit.add_command(label="Copy Hamiltonian (LaTeX)", command=lambda: copy_h("latex"))
    m_edit.add_command(label="Copy Hamiltonian (SymPy)", command=lambda: copy_h("sympy"))
    m_edit.add_command(label="Copy Commutators (LaTeX)",
                       command=lambda: copy_h("commutators"))
    menubar.add_cascade(label="Edit", menu=m_edit)

    m_act = tk.Menu(menubar, tearoff=0)
    m_act.add_command(label="Generate", command=generate, accelerator=f"{_MOD}↩ / F5")
    m_act.add_command(label="Dualize", command=dualize, accelerator=f"{_MOD}D")
    m_act.add_command(label="Diagonalize", command=diagonalize, accelerator=f"{_MOD}K")
    menubar.add_cascade(label="Actions", menu=m_act)

    # Circuits menu: load a ready-made circuit from the library
    def load_library(name):
        from . import library
        from .netlist import to_netlist
        try:
            ckt = library.CIRCUITS[name]()
            netlist.delete("1.0", "end")
            netlist.insert("1.0", to_netlist(ckt))
            clear_error()
            generate()
        except Exception as exc:
            report_error(exc)

    m_circ = tk.Menu(menubar, tearoff=0)
    from . import library as _lib
    for _name in _lib.CIRCUITS:
        m_circ.add_command(label=_name.replace("_", " ").title(),
                           command=lambda n=_name: load_library(n))
    menubar.add_cascade(label="Circuits", menu=m_circ)

    m_view = tk.Menu(menubar, tearoff=0)
    m_view.add_checkbutton(label="Energy units (E_C, E_L, n)",
                           variable=energy_units, command=lambda: _rerender())
    m_view.add_checkbutton(label="LaTeX (system TeX)",
                           variable=use_latex, command=lambda: _rerender())
    menubar.add_cascade(label="View", menu=m_view)
    root.config(menu=menubar)

    _restore_session()
    _dbg(f"main(): GUI built; python={__import__('sys').executable}")

    # Bring the window to the front and give it focus.  A terminal-launched
    # (non-bundled) Python app does not always become the macOS foreground app,
    # so clicks can be dropped until it is activated; focus_force + lift help,
    # and the keyboard shortcuts above are the reliable fallback.
    def _activate():
        try:
            root.lift()
            root.focus_force()
            netlist.focus_set()   # give the text box keyboard focus
        except Exception:
            pass
    root.after(80, _activate)
    root.after(400, _prewarm_fonts)
    # run the first generate inside the event loop (not before mainloop, which
    # can leave the macOS app unresponsive to clicks)
    root.after(100, generate)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    main()

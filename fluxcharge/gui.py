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

import os
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


def numerical_summary(netlist_text, params, n_levels=6, canonical=True):
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
    return summary_from_result(result, params, n_levels)


def summary_from_result(result, params, n_levels=6):
    """Build the numerical-summary dict from an already-reduced result.

    Lets the UI skip the (re)reduction when the circuit is unchanged since the
    last Generate -- diagonalization then costs only the matrix build + eigh.
    """
    modes = result.modes()
    ev = result.eigenenergies(params, n_levels=n_levels)
    return {
        "result": result,
        "modes": [(m.flux, m.charge, m.kind) for m in modes],
        "eigenenergies": ev,
        "transitions": [float(ev[i] - ev[0]) for i in range(1, len(ev))],
        "single_mode": len(modes) == 1,
    }


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


def _hamiltonian_latex(expr, operators):
    """LaTeX of *expr* with each operator symbol hatted (parameters left as
    ordinary c-numbers)."""
    names = {s: _operator_latex(s) for s in operators}
    return sp.latex(expr, symbol_names=names)


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
    import queue
    import threading
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import tkinter.font as tkfont
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
    left = ttk.Frame(body, width=470)
    left.pack(side="left", fill="y")
    left.pack_propagate(False)
    right = ttk.Frame(body)
    right.pack(side="left", fill="both", expand=True, padx=(12, 0))

    # ---- netlist card ----
    nl_card = card(left, "Netlist")
    nl_card.pack(fill="both", expand=True)
    netlist = tk.Text(nl_card, height=16, font=(MONO, 10), wrap="none")
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

    for txt, cmd in [("Load", do_load), ("Save", do_save), ("Example", do_example), ("Clear", do_clear)]:
        ttk.Button(fb, text=txt, command=cmd).pack(side="left", padx=(0, 4))
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

    gen_btn = ttk.Button(left, text="Generate  \u2192", style="Accent.TButton",
                         command=lambda: generate())
    gen_btn.pack(fill="x", pady=(12, 0))
    dual_btn = ttk.Button(left, text="Dualize  \u21c4   (LCG dual: C\u2194L, JJ\u2194QPS, G\u2192\u22121/G)",
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
    diag_btn = ttk.Button(num_card, text="Diagonalize", command=lambda: diagonalize())
    diag_btn.grid(row=1, column=2, rowspan=2, sticky="ns", padx=(4, 0))

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
    busy_state = {"n": 0}
    action_buttons = [gen_btn, dual_btn, diag_btn]

    def busy_on(text):
        busy_state["n"] += 1
        for btn in action_buttons:
            btn.state(["disabled"])
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
            for btn in action_buttons:
                btn.state(["!disabled"])

    def run_async(work, on_success, busy_text="computing…"):
        """Run *work()* (pure compute, no Tk/matplotlib) in a worker thread; call
        *on_success(result)* on the main thread when done.  Keeps the event loop
        free so the progress spinner animates and the window stays responsive."""
        busy_on(busy_text)
        box = {}

        def worker():
            try:
                box["ok"] = work()
            except Exception as exc:  # marshalled back to the main thread
                box["err"] = exc

        th = threading.Thread(target=worker, daemon=True)
        th.start()

        def check():
            if th.is_alive():
                root.after(60, check)
                return
            busy_off()
            if "err" in box:
                exc = box["err"]
                status.config(text=f"error: {exc}", foreground="#b00020")
                messagebox.showerror("fluxcharge", str(exc))
                return
            try:
                on_success(box["ok"])
            except Exception as exc:
                status.config(text=f"error: {exc}", foreground="#b00020")
                messagebox.showerror("fluxcharge", str(exc))

        root.after(60, check)

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
        ax_h.text(0.0, 0.95, label, transform=ax_h.transAxes, ha="left",
                  va="top", fontsize=9, color=MUTED, fontfamily="sans-serif")
        ax_h.text(0.5, 0.55, f"$\\hat{{H}} = {H_l}$",
                  ha="center", va="center", fontsize=16, color=INK)
        if e_units and out["energy_defs_latex"]:
            ax_h.text(0.5, 0.06, f"$({out['energy_defs_latex']})$",
                      ha="center", va="center", fontsize=9, color=MUTED)

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
        text = netlist.get("1.0", "end-1c")

        def work():
            # pure-sympy reduction off the main thread; no matplotlib here
            return compute(text, canonical=True, draw=False)

        def done(out):
            # the schematic uses schemdraw/matplotlib, so it must be drawn on the
            # main thread; show feedback because this part can briefly block
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
            _rerender()
            report.delete("1.0", "end")
            report.insert("end", f"H = {out['H']}\n\nLagrangian:\n"
                          f"{out['lagrangian']}\n\n{out['report']}")

        run_async(work, done, busy_text="reducing circuit…")

    def dualize():
        text = netlist.get("1.0", "end-1c")
        try:
            ckt = from_netlist(text)
            ckt.validate()
            d = dual(ckt)
            netlist.delete("1.0", "end")
            netlist.insert("1.0", to_netlist(d))
        except Exception as exc:
            status.config(text=f"dual error: {exc}", foreground="#b00020")
            messagebox.showerror("fluxcharge", str(exc))
            return
        generate()

    def diagonalize():
        from .__main__ import _parse_params
        text = netlist.get("1.0", "end-1c")
        try:
            params = _parse_params([params_entry.get()])
            n = max(1, int(levels_entry.get() or 6))
        except Exception as exc:
            status.config(text=f"diagonalize error: {exc}", foreground="#b00020")
            messagebox.showerror("fluxcharge", str(exc))
            return

        # reuse the reduction from the last Generate when the circuit is unchanged,
        # so diagonalizing costs only the matrix build + eigh (not a re-reduction)
        cached = (last["out"] if last["text"] == text and last["out"] else None)

        def work():
            if cached is not None:
                return summary_from_result(cached["result"], params, n_levels=n)
            return numerical_summary(text, params, n_levels=n)

        run_async(work, lambda summ: _show_diag(summ, params, n),
                  busy_text="diagonalizing…")

    def _show_diag(summ, params, n):
        ev = summ["eigenenergies"]
        lines = ["Mode types:"]
        for flux, charge, kind in summ["modes"]:
            lines.append(f"  {flux} / {charge}: {kind}")
        lines.append("\nEigenenergies:")
        lines += [f"  E_{i} = {e:.6g}" for i, e in enumerate(ev)]
        if summ["transitions"]:
            lines.append("transitions above ground: "
                         + ", ".join(f"{t:.5g}" for t in summ["transitions"]))
        report.delete("1.0", "end")
        report.insert("end", "\n".join(lines))
        status.config(text="diagonalized", foreground="#0a7d2c")

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
                summ["result"].plot_potential_wavefunctions(params, n_levels=n, ax=pax)
                drew = True
            except NotImplementedError:
                status.config(
                    text="no scalar potential (gyrator cross term); showing levels",
                    foreground="#b26a00")
        if not drew:
            try:
                summ["result"].plot_energy_levels(params, n_levels=n, ax=pax)
            except Exception as exc:
                pax.text(0.5, 0.5, f"plot unavailable:\n{exc}", ha="center",
                         va="center", wrap=True, fontsize=9)
        pcanvas = FigureCanvasTkAgg(pfig, master=win)
        pcanvas.get_tk_widget().pack(fill="both", expand=True)
        pcanvas.draw()

    # Pre-warm matplotlib's mathtext font cache off the main thread.  The first
    # equation render builds this cache and can take several seconds; doing it
    # here (with a non-Tk Agg canvas, so it is thread-safe) means the first real
    # render is fast instead of freezing the window right after the first click.
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

    threading.Thread(target=_prewarm_fonts, daemon=True).start()

    generate()
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    main()

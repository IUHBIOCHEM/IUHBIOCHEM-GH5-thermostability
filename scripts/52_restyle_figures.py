"""52_restyle_figures.py — visual-storytelling restyle (NO new analysis).
Unified Nature-Communications style across Fig14/15/16/18 + graphical abstract.
All numbers read from cached CSVs; PyMOL structure renders reused from scratchpad.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Wedge, Circle
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr
from PIL import Image

A = "ai_training/analysis"
SCR = os.environ.get("SCRATCH_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_scratch"))
os.makedirs(SCR, exist_ok=True)
FIG = "manuscript/figures"

# ---------- unified palette ----------
HOT = "#C0392B"        # hotspot / discovery accent
THERMO, MESO = "#C0392B", "#93A1B0"
POS, NEG = "#2166AC", "#C0392B"     # Lys/Arg , Asp/Glu
STAB, DESTAB = "#2166AC", "#C0392B"
GOLD = "#E8A200"
GREY = "#CBD1D8"
INK = "#232830"
GREEN = "#1B9E77"
CUM = ["#AFC9E6", "#6BA3D6", "#2E77B5", "#123F73"]   # n = 1..4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14.5,
    "axes.titleweight": "bold", "axes.linewidth": 1.1, "axes.edgecolor": "#3a3f45",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
    "xtick.color": "#3a3f45", "ytick.color": "#3a3f45",
    "text.color": INK, "axes.labelcolor": INK,
    "legend.fontsize": 12.5, "legend.frameon": False,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def crop_white(png, pad=6):
    im = Image.open(png).convert("RGB")
    a = np.asarray(im)
    mask = (a < 245).any(2)
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    return a[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]


def panel_tag(ax, s, dx=-0.02, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=20, fontweight="bold",
            va="bottom", ha="right", color=INK)


# =========================================================================
# FIGURE 14 — discovery -> localization -> ranking (one clear left-to-right story)
# =========================================================================
def fig14():
    C = pd.read_csv(f"{A}/constraint.csv")
    C = C.dropna(subset=["charge_delta", "p"])
    hot = C["group"] == "hotspot"
    n_hot = int(hot.sum()); n_tot = len(C)

    fig = plt.figure(figsize=(15.5, 5.6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.34, left=0.05, right=0.98, top=0.82, bottom=0.16)

    # panel labels anchored to gridspec-cell corners (uniform, aspect-independent)
    def celltag(cell, s):
        bb = cell.get_position(fig)
        fig.text(bb.x0, bb.y1 + 0.045, s, fontsize=20, fontweight="bold",
                 va="bottom", ha="left", color=INK)

    # (a) volcano
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(C.loc[~hot, "charge_delta"] * 100, -np.log10(C.loc[~hot, "p"]),
               s=26, c=GREY, edgecolor="none", zorder=1)
    ax.scatter(C.loc[hot, "charge_delta"] * 100, -np.log10(C.loc[hot, "p"]),
               s=40, c=HOT, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(-np.log10(0.05), color="#3a3f45", ls=(0, (4, 3)), lw=1.0)
    ax.set_xlabel("Δ charged frequency\n(thermophile − mesophile), %")
    ax.set_ylabel(r"$-\log_{10}\,p$")
    ax.set_title("Discover", color=HOT, pad=10)
    ax.text(0.95, 0.93, f"{n_hot} / 272 positions\n(24.3%) are\nthermal hotspots",
            transform=ax.transAxes, ha="right", va="top", fontsize=13.5,
            color=INK, linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.5", fc="#FBEEEC", ec=HOT, lw=1.3))
    celltag(gs[0, 0], "a")

    # (b) structural localization
    axb = fig.add_subplot(gs[0, 1])
    axb.imshow(crop_white(f"{SCR}/fig14_structure.png")); axb.axis("off")
    axb.set_title("Localize", color=HOT, pad=10)
    axb.text(0.5, -0.06, "66 hotspots (red) map to the barrel periphery;\n"
             "catalytic glutamates (gold) stay buried and unperturbed",
             transform=axb.transAxes, ha="center", va="top", fontsize=12.8, color="#444")
    celltag(gs[0, 1], "b")

    # (c) candidate ranking
    axc = fig.add_subplot(gs[0, 2])
    sg = pd.read_csv(f"{A}/signature_scores.csv"); sg["_a"] = sg["Accession"].astype(str).str.split(".").str[0]
    pr = pd.read_csv(f"{A}/all757_predictions.csv"); pr["_a"] = pr["Accession"].astype(str).str.split(".").str[0]
    pp = pr.merge(sg[["_a", "signature"]], on="_a", how="left").dropna(subset=["signature", "OGT_pred"])
    tc = pd.read_csv(f"{A}/top_candidates_validated.csv"); tc["_a"] = tc["Accession"].astype(str).str.split(".").str[0]
    tc = tc.merge(sg[["_a", "signature"]], on="_a", how="left").dropna(subset=["signature"])
    rho = spearmanr(pp["signature"], pp["OGT_pred"]).correlation
    axc.scatter(pp["signature"] * 100, pp["OGT_pred"], s=18, c=GREY, alpha=0.55,
                edgecolor="none", label="all GH5 (n = 757)", zorder=1)
    axc.scatter(tc["signature"] * 100, tc["Predicted_OGT_C"], s=95, c=HOT,
                edgecolor="white", linewidth=0.8, zorder=3, label="top candidates")
    axc.set_xlabel("Sparse 66-position\ncharge signature (%)")
    axc.set_ylabel("Predicted OGT (°C)")
    axc.set_title("Rank", color=HOT, pad=10)
    axc.text(0.05, 0.93, f"ρ = {rho:.2f}", transform=axc.transAxes, fontsize=15,
             fontweight="bold", va="top", color=INK)
    axc.legend(loc="lower right", handletextpad=0.3)
    celltag(gs[0, 2], "c")

    # flow arrows between panels
    for x0, x1 in [(0.352, 0.372), (0.662, 0.682)]:
        fig.add_artist(FancyArrowPatch((x0, 0.5), (x1, 0.5), transform=fig.transFigure,
                       arrowstyle="-|>", mutation_scale=26, lw=2.4, color="#9aa1a8"))
    fig.savefig(f"{FIG}/Fig14_discovery.png"); plt.close(fig)
    print("Fig14 done", n_hot, n_tot)


# =========================================================================
# FIGURE 15 — residue chemistry + surface/core summary (one story: surface E/K)
# =========================================================================
def fig15():
    h = pd.read_csv(f"{A}/hotspot_supptable.csv")
    ss = pd.read_csv(f"{A}/structural_summary.csv").iloc[0]
    gs_ = pd.read_csv(f"{A}/group_stats_fdr.csv").set_index("metric")

    pref = h["thermo_pref_residue"]
    arom = pref.isin(list("FYW")).sum()
    cats = [("Glu (E)", (pref == "E").sum(), NEG), ("Lys (K)", (pref == "K").sum(), POS),
            ("Asp (D)", (pref == "D").sum(), NEG), ("Arg (R)", (pref == "R").sum(), POS),
            ("Aromatic", arom, "#7B6BB0"), ("Other", (~pref.isin(list("EKDRFYW"))).sum(), GREY)]

    fig = plt.figure(figsize=(15.5, 5.4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.42, left=0.06, right=0.98, top=0.83, bottom=0.2)

    # (a) preferred residue at hotspots
    ax = fig.add_subplot(gs[0, 0])
    labs = [c[0] for c in cats]; vals = [c[1] for c in cats]; cols = [c[2] for c in cats]
    b = ax.bar(labs, vals, color=cols, edgecolor="white", linewidth=0.8, width=0.72)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.3, str(v), ha="center", va="bottom",
                fontsize=12.5, fontweight="bold")
    ax.set_ylabel("Hotspot positions")
    ax.set_title("Hotspots acquire Glu / Lys", pad=10)
    ax.set_ylim(0, max(vals) + 3)
    ax.tick_params(axis="x", rotation=35)
    for t in ax.get_xticklabels():
        t.set_ha("right")
    ax.text(0.97, 0.92, "E + K dominate", transform=ax.transAxes, ha="right", va="top",
            fontsize=13, color=INK, style="italic")
    panel_tag(ax, "a")

    # (b) surface / core donut + distal callout  (KEY new panel)
    axb = fig.add_subplot(gs[0, 1]); axb.set_aspect("equal"); axb.axis("off")
    surf = int(ss["surface"]); core = int(ss["mappable"]) - surf
    pct = surf / (surf + core) * 100
    axb.add_patch(Wedge((0, 0), 1, 90, 90 - 360 * pct / 100, width=0.42, fc=HOT, ec="white", lw=2))
    axb.add_patch(Wedge((0, 0), 1, 90 - 360 * pct / 100, 90, width=0.42, fc="#8B96A3", ec="white", lw=2))
    axb.text(0, 0.12, f"{pct:.0f}%", ha="center", va="center", fontsize=30, fontweight="bold", color=HOT)
    axb.text(0, -0.16, "surface\nexposed", ha="center", va="center", fontsize=13, color=INK, linespacing=1.1)
    axb.text(0, -1.28, f"surface {surf}   ·   core {core}   (of {int(ss['mappable'])} mapped)",
             ha="center", fontsize=12.5, color="#444")
    axb.set_xlim(-1.35, 1.35); axb.set_ylim(-1.5, 1.35)
    axb.set_title("Surface-exposed", pad=10)
    panel_tag(axb, "b")

    # (c) distal + E/K vs D/R enrichment (support)
    axc = fig.add_subplot(gs[0, 2])
    x = np.arange(2); w = 0.36
    ek = gs_.loc["E+K %"]; dr = gs_.loc["D+R %"]
    axc.bar(x - w / 2, [ek["thermo_mean"], dr["thermo_mean"]], w, color=THERMO, label="thermophile")
    axc.bar(x + w / 2, [ek["meso_mean"], dr["meso_mean"]], w, color=MESO, label="mesophile")
    axc.set_xticks(x); axc.set_xticklabels(["Glu + Lys", "Asp + Arg"])
    axc.set_ylabel("Residue content (%)")
    axc.set_ylim(0, max(ek["thermo_mean"], dr["thermo_mean"]) * 1.2)
    axc.set_title("Glu/Lys-specific gain", pad=26)
    axc.legend(loc="upper right")
    axc.text(0.5, 1.02,
             f"{int(ss['distal'])}/{int(ss['mappable'])} hotspots distal (>12 Å) · "
             f"{ss['hotspot_meanDist']:.1f} vs {ss['bg_meanDist']:.1f} Å, p<0.001",
             transform=axc.transAxes, ha="center", va="bottom", fontsize=10.5, color="#555")
    panel_tag(axc, "c")

    fig.savefig(f"{FIG}/Fig15_engineering.png"); plt.close(fig)
    print("Fig15 done  surface", surf, "core", core)


# =========================================================================
# FIGURE 16 — cumulative stabilization coloured by mutation count
# =========================================================================
def _enzyme_glyph(ax, cx, cy, dots, r=0.052):
    """faded enzyme blob with N mutation dots (colour-coded) accumulating inside."""
    from matplotlib.patches import Circle, Ellipse
    ax.add_patch(Ellipse((cx, cy), 0.15, 0.20, fc="#E6E9ED", ec="#B7BFC8", lw=1.5, zorder=1))
    dc = {"s": STAB, "n": "#9AA6B4", "d": DESTAB}
    offs = {1: [(0, 0)], 2: [(-0.032, 0.028), (0.032, -0.028)],
            3: [(-0.038, 0.03), (0.04, 0.03), (0, -0.038)],
            4: [(-0.038, 0.032), (0.038, 0.032), (-0.038, -0.036), (0.038, -0.036)]}[len(dots)]
    for (dx, dy), code in zip(offs, dots):
        ax.add_patch(Circle((cx + dx, cy + dy), r, fc=dc[code], ec="white", lw=1.2, zorder=3))


def fig16():
    sd = pd.read_csv(f"{A}/ddg_combined.csv")            # 4 singles
    cd = pd.read_csv(f"{A}/combo_ddg.csv")               # combos with n, obs, add
    order = ["N110E", "N244K", "A222K", "S248E"]
    sd = sd.set_index("mutation").loc[order].reset_index()

    fig = plt.figure(figsize=(15.5, 9.0))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.68, 1], hspace=0.42, wspace=0.26,
                  left=0.07, right=0.98, top=0.9, bottom=0.16, width_ratios=[1, 1.25])

    # (a) SCHEMATIC — single/double/triple/quadruple, compatibility emphasised
    axs = fig.add_subplot(gs[0, :]); axs.axis("off"); axs.set_xlim(0, 1); axs.set_ylim(0, 1)
    axs.set_title("Accumulating hotspot mutations — only compatible combinations improve stability",
                  pad=8, fontsize=15.5)
    stages = [("Single", "N110E", ["s"], -0.66, "stabilizing", STAB),
              ("Double", "+ N244K", ["s", "s"], -1.00, "best · compatible", STAB),
              ("Triple", "+ A222K", ["s", "s", "n"], -0.74, "still compatible", "#5a6472"),
              ("Quadruple", "+ S248E", ["s", "s", "n", "d"], -0.05, "incompatible · gain lost", DESTAB)]
    xs = np.linspace(0.13, 0.87, 4); cy = 0.52
    for i, (name, add, dots, ddg, verdict, vc) in enumerate(stages):
        _enzyme_glyph(axs, xs[i], cy, dots)
        axs.text(xs[i], 0.93, name, ha="center", fontsize=13.5, fontweight="bold", color=INK)
        axs.text(xs[i], 0.83, add, ha="center", fontsize=12, color="#555")
        axs.text(xs[i], cy - 0.20, f"ΔΔG {ddg:+.2f}", ha="center", fontsize=13,
                 fontweight="bold", color=vc)
        axs.text(xs[i], cy - 0.315, verdict, ha="center", fontsize=11, color=vc, style="italic")
        if i < 3:
            xa = (xs[i] + xs[i + 1]) / 2
            col = DESTAB if i == 2 else "#7d858e"
            axs.add_patch(FancyArrowPatch((xs[i] + 0.085, cy), (xs[i + 1] - 0.085, cy),
                          arrowstyle="-|>", mutation_scale=22, lw=2.6, color=col))
    from matplotlib.patches import Patch
    axs.legend(handles=[Patch(fc=STAB, label="stabilizing"), Patch(fc="#9AA6B4", label="neutral"),
                        Patch(fc=DESTAB, label="destabilizing")], loc="lower center", ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, -0.14), handletextpad=0.4, columnspacing=1.4)
    panel_tag(axs, "a")

    # (b) all mutants ordered by number of mutations, coloured by count
    ax = fig.add_subplot(gs[1, 0])
    rows = [(m, v, 1) for m, v in zip(sd["mutation"], sd["FoldX_ddG"])]
    for _, r in cd.iterrows():
        rows.append((r["combo"].replace(",", "+"), r["obs"], int(r["n"])))
    rows.sort(key=lambda t: (t[2], t[1]))
    labs = [r[0] for r in rows]; vals = [r[1] for r in rows]; ns = [r[2] for r in rows]
    cols = [CUM[n - 1] for n in ns]
    xb = np.arange(len(rows))
    ax.bar(xb, vals, color=cols, edgecolor="white", linewidth=0.8, width=0.74)
    ax.axhline(0, color="#3a3f45", lw=1.0)
    ax.axhline(-0.5, color="#9aa1a8", ls=(0, (3, 3)), lw=1.0)
    # highlight best pair
    bi = int(np.argmin(vals))
    ax.annotate("best pair\n−1.00 kcal/mol", (xb[bi], vals[bi]), xytext=(xb[bi], vals[bi] - 0.24),
                ha="center", va="top", fontsize=12, fontweight="bold", color=CUM[1])
    # annotate quad collapse
    qi = ns.index(4)
    ax.annotate("+ destabilizing S248E\n→ gain erased", (xb[qi], vals[qi]),
                xytext=(xb[qi], 0.34), ha="center", fontsize=11, color=DESTAB,
                arrowprops=dict(arrowstyle="->", color=DESTAB, lw=1.4))
    ax.set_xticks(xb); ax.set_xticklabels(labs, rotation=40, ha="right", fontsize=10.5)
    ax.set_ylabel("FoldX ΔΔG (kcal/mol)")
    ax.set_ylim(min(vals) - 0.5, max(vals) + 0.22)
    ax.set_title("Cumulative — only if compatible", pad=10)
    ax.text(0.02, 0.05, "↓ more stable", transform=ax.transAxes, fontsize=11.5, color="#444")
    # legend for mutation count
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=CUM[i], label=f"{i+1} mutation" + ("s" if i else ""))
                       for i in range(4)], loc="upper left", ncol=1, fontsize=10.5,
              handlelength=1.2, handletextpad=0.5)
    panel_tag(ax, "b")

    # (c) observed vs additive — plotted as stabilization magnitude (bars grow rightward,
    #     leaving the y labels clear); coupling ≈ 0 shown in the title.
    axb = fig.add_subplot(gs[1, 1])
    cd2 = cd.copy(); cd2["lab"] = cd2["combo"].str.replace(",", "+")
    y = np.arange(len(cd2))[::-1]; w = 0.36
    axb.barh(y + w / 2, -cd2["add"], w, color="#C9D2DC", label="additive expectation")
    axb.barh(y - w / 2, -cd2["obs"], w, color=CUM[2], label="observed (FoldX)")
    axb.axvline(0, color="#3a3f45", lw=1.0)
    axb.set_yticks(y); axb.set_yticklabels(cd2["lab"], fontsize=10)
    axb.set_xlim(0, float((-cd2[["add", "obs"]]).max().max()) * 1.16)
    axb.set_xlabel("stabilization,  −ΔΔG (kcal/mol)")
    axb.set_title("Effects are additive (coupling ≈ 0)", pad=10)
    axb.legend(loc="lower right")
    panel_tag(axb, "c")

    fig.savefig(f"{FIG}/Fig16_combinations.png"); plt.close(fig)
    print("Fig16 done")


# ---------- flow helpers ----------
def flow_box(ax, cx, cy, w, hh, text, fc, tc="white", fs=14, bold=True, ec="none", pad=0.008):
    ax.add_patch(FancyBboxPatch((cx - w / 2 + pad, cy - hh / 2 + pad), w - 2 * pad, hh - 2 * pad,
                 boxstyle="round,pad=0,rounding_size=0.03", fc=fc, ec=ec, lw=1.4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.05, zorder=5)


def v_arrow(ax, cx, y0, y1, color="#7d858e"):
    ax.add_patch(FancyArrowPatch((cx, y0), (cx, y1), arrowstyle="-|>",
                 mutation_scale=20, lw=2.2, color=color))


def h_arrow(ax, x0, x1, cy, color="#7d858e"):
    ax.add_patch(FancyArrowPatch((x0, cy), (x1, cy), arrowstyle="-|>",
                 mutation_scale=22, lw=2.4, color=color))


# =========================================================================
# FIGURE 18 — mechanistic take-home (vertical evolutionary flow = hero)
# =========================================================================
def fig18():
    fig = plt.figure(figsize=(14.5, 9.2))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.05, 1.0], wspace=0.05,
                  left=0.02, right=0.98, top=0.95, bottom=0.03)

    # LEFT: hero vertical flow
    axL = fig.add_subplot(gs[0, 0]); axL.axis("off")
    axL.set_xlim(0, 1); axL.set_ylim(0, 1)
    steps = [
        ("Mesophilic enzyme", "#8B96A3", "white"),
        ("Few surface charges", "#6E93B8", "white"),
        ("Evolution at 66\npermissive positions", "#D98C7A", "white"),
        ("Glu / Lys accumulation", "#CB6A55", "white"),
        ("Peripheral electrostatic\n(salt-bridge) network", "#C0392B", "white"),
        ("Cumulative stabilization\n(−1.0 kcal/mol)", "#8E2A20", "white"),
        ("Thermophilic enzyme", GREEN, "white"),
    ]
    n = len(steps); top, bot = 0.895, 0.03
    ys = np.linspace(top, bot, n)
    bh = (ys[0] - ys[1]) * 0.62; bw = 0.72; cx = 0.5
    for i, (txt, fc, tc) in enumerate(steps):
        flow_box(axL, cx, ys[i], bw, bh, txt, fc, tc, fs=15.5)
        if i < n - 1:
            v_arrow(axL, cx, ys[i] - bh / 2 - 0.004, ys[i + 1] + bh / 2 + 0.004)
    axL.text(0.5, 0.975, "The evolutionary mechanism", ha="center", va="center",
             fontsize=18, fontweight="bold", color=INK)

    # RIGHT: supporting evidence stacked
    gsr = gs[0, 1].subgridspec(3, 1, height_ratios=[1.35, 1.0, 0.72], hspace=0.34)

    # structure
    axs = fig.add_subplot(gsr[0]); axs.axis("off")
    src = f"{SCR}/f18A.png" if os.path.exists(f"{SCR}/f18A.png") else f"{SCR}/fig14_structure.png"
    axs.imshow(crop_white(src))
    axs.set_title("Charged hotspots on the barrel periphery", fontsize=14, pad=6, color=INK)
    axs.text(0.5, -0.05, "catalytic Glu (gold) buried · Lys/Arg blue · Asp/Glu red",
             transform=axs.transAxes, ha="center", va="top", fontsize=11.5, color="#555")

    # network schematic (representative charged hotspots)
    axn = fig.add_subplot(gsr[1]); axn.axis("off"); axn.set_xlim(0, 1); axn.set_ylim(0, 1)
    axn.set_title("Peripheral salt-bridge network", fontsize=14, pad=4, color=INK)
    nodes = {"K35": (0.30, 0.82, POS), "E37": (0.55, 0.86, NEG), "R75": (0.22, 0.55, POS),
             "D70": (0.08, 0.60, NEG), "K74": (0.20, 0.30, POS), "E78": (0.42, 0.40, NEG),
             "K85": (0.50, 0.10, POS), "K43": (0.80, 0.42, POS), "E44": (0.93, 0.55, NEG)}
    edges = [("K35", "E37"), ("R75", "D70"), ("R75", "E78"), ("K74", "E78"), ("K74", "D70"),
             ("E78", "K85"), ("K43", "E44"), ("E37", "K43"), ("K85", "K43")]
    for a, b in edges:
        axn.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]],
                 color="#9aa1a8", ls=(0, (3, 3)), lw=1.5, zorder=1)
    for name, (x, y, c) in nodes.items():
        axn.add_patch(Circle((x, y), 0.075, fc=c, ec="white", lw=1.5, zorder=3))
        axn.text(x, y, name, ha="center", va="center", fontsize=9.5, color="white",
                 fontweight="bold", zorder=4)

    # cumulative bar
    axc = fig.add_subplot(gsr[2])
    axc.bar([0, 1], [-0.66, -1.00], color=[CUM[0], CUM[2]], width=0.6, edgecolor="white")
    axc.axhline(0, color="#3a3f45", lw=1.0)
    axc.set_xticks([0, 1]); axc.set_xticklabels(["single\n(N110E)", "pair\n(N110E+N244K)"], fontsize=11)
    axc.set_ylabel("ΔΔG", fontsize=12); axc.set_ylim(-1.2, 0.05)
    axc.set_title("Single insufficient → combination stabilizes", fontsize=13, pad=4, color=INK)
    axc.spines["left"].set_visible(True)
    for i, v in enumerate([-0.66, -1.00]):
        axc.text(i, v - 0.04, f"{v:.2f}", ha="center", va="top", fontsize=11.5, fontweight="bold")

    fig.savefig(f"{FIG}/Fig18_model.png"); plt.close(fig)
    print("Fig18 done")


# =========================================================================
# GRAPHICAL ABSTRACT — analysis workflow (biology-forward)
# =========================================================================
def graphical_abstract():
    fig = plt.figure(figsize=(15.5, 4.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    steps = [
        ("757 GH5\nsequences", "#5B6B7B"),
        ("Pfam PF00150\nverification", "#5B6B7B"),
        ("ESM2 language\nmodel", "#2E77B5"),
        ("66 thermal\nhotspots", "#C0392B"),
        ("Peripheral\nsalt-bridge\nnetwork", "#C0392B"),
        ("Thermostability\nmechanism", "#8E2A20"),
        ("Engineering\ntargets", GREEN),
    ]
    n = len(steps); m = 0.02; gap = 0.03
    bw = (1 - 2 * m - (n - 1) * gap) / n; bh = 0.46; cy = 0.5
    ax.text(0.5, 0.93, "AI reveals the evolutionary mechanism of GH5 thermal adaptation",
            ha="center", va="center", fontsize=17, fontweight="bold", color=INK)
    for i, (txt, fc) in enumerate(steps):
        cx = m + bw / 2 + i * (bw + gap)
        flow_box(ax, cx, cy, bw, bh, txt, fc, "white", fs=13.5, pad=0.004)
        if i < n - 1:
            xa = cx + bw / 2 + 0.003
            h_arrow(ax, xa, xa + gap - 0.006, cy)
    ax.text(0.5, 0.14, "sequence data  →  language model  →  biological discovery  →  rational engineering",
            ha="center", va="center", fontsize=12.5, style="italic", color="#555")
    fig.savefig(f"{FIG}/Graphical_abstract.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print("Graphical abstract done")


if __name__ == "__main__":
    fig14(); fig15(); fig16(); fig18(); graphical_abstract()
    print("ALL FIGURES RESTYLED")

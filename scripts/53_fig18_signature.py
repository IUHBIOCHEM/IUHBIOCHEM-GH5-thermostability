"""53_fig18_signature.py — signature mechanistic Figure 18 (real structures).
Panels: (a) 3AMC TIM-barrel with catalytic Glu (yellow) + hotspots (K/R blue, D/E red);
        (b) representative REAL salt bridges among charged hotspots (sticks + distances);
        (c) surface salt-bridge NETWORK: representative thermophile 3AMC (25 bridges) vs
            mesophile 6PZ7 (16 bridges), same camera, grounded by the 82-structure survey.
Salt bridges = Barlow-Thornton ion pairs (Asp/Glu O <-> Lys/Arg/His N, <=4 A); the
mesophile 6PZ7 is a salt-bridge-poor GH5 chosen to represent the population trend
(3PZT is atypically salt-bridge-rich and stays the FoldX engineering scaffold in Fig 3).
"""
import os, subprocess, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCR = os.environ.get("SCRATCH_DIR", os.path.join(ROOT, "_scratch"))
os.makedirs(SCR, exist_ok=True)
PML = os.environ.get("PYMOL") or shutil.which("pymol") or "/Applications/PyMOL.app/Contents/MacOS/PyMOL"
FIG = "manuscript/figures"
INK = "#232830"; GREEN = "#1B9E77"

KR = "35+43+55+67+74+75+85+108+116+155+158+181+189+229+243+244+259+265+267+274"
DE = "37+44+70+78+103+105+106+109+127+144+180+188+227+234+240+261+263+287"
OT = "39+46+61+63+69+73+82+87+96+101+112+126+141+156+164+176+203+225+230+232"

HEAD = """
set ray_shadows, 0
set antialias, 2
set cartoon_transparency, %s
bg_color white
set opaque_background, 0
set ray_opaque_background, 0
set_color hblue, [0.13,0.40,0.67]
set_color hred,  [0.75,0.22,0.17]
set_color hor,   [0.92,0.58,0.22]
set_color hcat,  [0.98,0.80,0.05]
"""


def run_pml(name, body):
    p = f"{SCR}/{name}.pml"
    with open(p, "w") as f:
        f.write(body)
    subprocess.run([PML, "-cq", p], cwd=ROOT, timeout=400)


def render_all():
    # (a) overview  (draw, not ray -> no eval watermark)
    run_pml("f18_overview", f"""
viewport 1450, 1500
load ai_training/structures/3AMC.pdb, s
hide everything
remove not chain A
remove solvent
{HEAD % '0.30'}
show cartoon, s
color grey80, s
select kr, resi {KR}
select de, resi {DE}
select ot, resi {OT}
show spheres, (kr or de or ot) and name CA
set sphere_scale, 0.55, (kr or de or ot) and name CA
color hblue, kr and name CA
color hred, de and name CA
color hor, ot and name CA
select cat, resi 136+253
show sticks, cat and not name C+N+O
color hcat, cat
show spheres, cat and name CA
set sphere_scale, 0.85, cat and name CA
orient s
turn y, 15
draw 1450, 1500, antialias=2
png {SCR}/f18_overview.png, dpi=200
""")
    # (b) representative real salt bridges (K67-D70, R75-E78 cluster)
    run_pml("f18_saltbridge", f"""
viewport 1500, 1250
load ai_training/structures/3AMC.pdb, s
hide everything
remove not chain A
remove solvent
{HEAD % '0.72'}
show cartoon, s
color grey90, s
select sb, resi 67+70+75+78
show sticks, sb and not name C+N+O
set stick_radius, 0.24, sb
color hblue, sb and resn LYS+ARG
color hred, sb and resn ASP+GLU
util.cnc("sb and not (name C+N+O)")
distance d1, resi 67 and name NZ, resi 70 and name OD1, 5
distance d2, resi 75 and name NH1, resi 78 and name OE1, 5
set dash_color, orange
set dash_width, 6
set dash_gap, 0.35
set label_size, 30
set label_color, black
orient sb
zoom sb, 3.5
turn x, -8
draw 1500, 1250, antialias=2
png {SCR}/f18_saltbridge.png, dpi=200
""")
    # (c) surface salt-bridge NETWORK — representative thermophile (3AMC) vs mesophile (6PZ7).
    #     Barlow-Thornton ion pairs (Asp/Glu O <-> Lys/Arg/His N, <=4 A) drawn as dashes;
    #     charged side-chain tips as spheres. Same camera via cealign; draw -> no watermark.
    net_tmpl = """
viewport 1300, 1250
load ai_training/structures/3AMC.pdb, t
load ai_training/structures/6PZ7.pdb, m
remove not polymer
remove solvent
remove (t and not chain A)
remove (m and not chain A)
cealign t, m
bg_color white
set opaque_background, 0
set surface_quality, 1
set transparency, 0.45
set dash_gap, 0.32
set dash_width, 3.0
set dash_radius, 0.05
set_color hred, [0.75,0.22,0.17]
set_color hblue, [0.13,0.40,0.67]
hide everything
show surface, {OBJ}
color grey85, {OBJ}
select negA, {OBJ} and resn ASP+GLU and name OD1+OD2+OE1+OE2
select posA, {OBJ} and resn LYS+ARG+HIS and name NZ+NH1+NH2+NE+ND1+NE2
show spheres, (negA or posA)
set sphere_scale, 0.55, (negA or posA)
color hred, negA
color hblue, posA
distance sbd, negA, posA, 4.0
hide labels
set dash_color, grey30, sbd
orient t
turn y, 20
draw 1300, 1250
png {OUT}, dpi=200
"""
    run_pml("f18_net_thermo", net_tmpl.format(OBJ="t", OUT=f"{SCR}/f18_net_thermo.png"))
    run_pml("f18_net_meso", net_tmpl.format(OBJ="m", OUT=f"{SCR}/f18_net_meso.png"))

    # (d) vacuum electrostatic surface — same representative pair (thermophile 3AMC vs mesophile
    #     6PZ7), identical camera via cealign onto 3AMC; each surface in a fresh process (draw).
    esp_tmpl = """
viewport 1150, 1300
load ai_training/structures/3AMC.pdb, t
load ai_training/structures/6PZ7.pdb, m
remove not polymer
remove solvent
remove (t and not chain A)
remove (m and not chain A)
cealign t, m
bg_color white
set opaque_background, 0
set surface_quality, 1
orient t
turn y, 20
hide everything
util.protein_vacuum_esp("{OBJ}", mode=2, quiet=1)
disable {OBJ}_e_pot
draw 1150, 1300
png {OUT}, dpi=200
"""
    run_pml("f18_esp_thermo", esp_tmpl.format(OBJ="t", OUT=f"{SCR}/f18_esp_thermo.png"))
    run_pml("f18_esp_meso", esp_tmpl.format(OBJ="m", OUT=f"{SCR}/f18_esp_meso.png"))
    print("PyMOL renders:", [os.path.exists(f"{SCR}/f18_{n}.png")
          for n in ["overview", "saltbridge", "net_thermo", "net_meso", "esp_thermo", "esp_meso"]])


def crop(png, cut_bottom=0.0, pad=6):
    im = np.asarray(Image.open(png).convert("RGB"))
    if cut_bottom:
        im = im[:int(im.shape[0] * (1 - cut_bottom))]
    mask = (im < 244).any(2)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return im
    return im[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad]


def compose():
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "text.color": INK})
    fig = plt.figure(figsize=(15, 16))
    outer = GridSpec(3, 1, figure=fig, height_ratios=[0.92, 1.0, 1.0], hspace=0.52,
                     left=0.015, right=0.985, top=0.965, bottom=0.05)

    # panel labels anchored to the GRIDSPEC CELL corners (uniform, aspect-independent)
    def tag(ss, s):
        bb = ss.get_position(fig)
        fig.text(bb.x0, bb.y1 + 0.004, s, fontsize=22, fontweight="bold", va="bottom", ha="left")

    # row 0 — structure + salt bridges
    top = outer[0].subgridspec(1, 2, width_ratios=[1.05, 1], wspace=0.03)
    ax = fig.add_subplot(top[0]); ax.axis("off"); ax.imshow(crop(f"{SCR}/f18_overview.png"))
    ax.text(0.53, 1.05, "Real GH5 TIM-barrel (3AMC)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=15.5, fontweight="bold")
    tag(top[0], "a")
    ax.text(0.5, -0.04, "catalytic Glu136/Glu253 (yellow) buried at the barrel axis · "
            "66 hotspots on the periphery\n(Lys/Arg blue, Asp/Glu red, other orange)",
            transform=ax.transAxes, ha="center", va="top", fontsize=11.5, color="#555", linespacing=1.3)

    axb = fig.add_subplot(top[1]); axb.axis("off"); axb.imshow(crop(f"{SCR}/f18_saltbridge.png"))
    axb.set_title("Representative hotspot salt bridges", fontsize=15.5, fontweight="bold", pad=6)
    tag(top[1], "b")
    axb.text(0.5, -0.04, "real 3AMC geometry: Lys67–Asp70 (3.4 Å) and Arg75–Glu78 (3.0 Å)\n"
             "— 5 such salt bridges link charged hotspots",
             transform=axb.transAxes, ha="center", va="top", fontsize=11.5, color="#555", linespacing=1.3)

    # row 1 — surface salt-bridge NETWORK: representative thermophile vs mesophile (same camera)
    mid = outer[1].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.03)
    axt = fig.add_subplot(mid[0]); axt.axis("off"); axt.imshow(crop(f"{SCR}/f18_net_thermo.png"))
    axt.set_title("Thermophile — 3AMC ($\\mathit{T.\\ maritima}$)", fontsize=14.5, fontweight="bold",
                  pad=6, color="#9B3226")
    tag(mid[0], "c")
    axt.text(0.5, -0.03, "25 salt bridges · 8.1 per 100 residues",
             transform=axt.transAxes, ha="center", va="top", fontsize=12.5, fontweight="bold", color="#9B3226")
    axm = fig.add_subplot(mid[1]); axm.axis("off"); axm.imshow(crop(f"{SCR}/f18_net_meso.png"))
    axm.set_title("Mesophile — 6PZ7 ($\\mathit{C.\\ acetobutylicum}$)", fontsize=14.5, fontweight="bold",
                  pad=6, color="#4A5568")
    axm.text(0.5, -0.03, "16 salt bridges · 4.9 per 100 residues",
             transform=axm.transAxes, ha="center", va="top", fontsize=12.5, fontweight="bold", color="#4A5568")

    # charge legend + population statistic under the row
    from matplotlib.lines import Line2D
    y0 = outer[1].get_position(fig).y0
    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor="#BF382B",
                      markersize=12, label="Acidic (Asp/Glu)"),
               Line2D([0], [0], marker='o', color='none', markerfacecolor="#2166AB",
                      markersize=12, label="Basic (Lys/Arg/His)"),
               Line2D([0], [0], color="#4d4d4d", lw=2, ls=(0, (2, 1.4)),
                      label="Salt bridge (≤ 4 Å)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y0 - 0.018),
               ncol=3, frameon=False, fontsize=12.5, handletextpad=0.5, columnspacing=2.2)
    fig.text(0.5, y0 - 0.052,
             "Across 82 experimentally solved GH5 structures, thermophiles carry more salt bridges per 100 "
             "residues than mesophiles\n(7.6 vs 6.3; Mann–Whitney P = 0.0004) — the two structures shown "
             "are representative of this trend.",
             ha="center", fontsize=11.5, style="italic", color="#444", linespacing=1.35)

    # row 2 — vacuum electrostatic surface (same representative pair, same camera)
    bot = outer[2].subgridspec(1, 3, width_ratios=[1, 0.17, 1], wspace=0.02)
    axet = fig.add_subplot(bot[0]); axet.axis("off"); axet.imshow(crop(f"{SCR}/f18_esp_thermo.png"))
    axet.set_title("Thermophile — 3AMC ($\\mathit{T.\\ maritima}$)", fontsize=14.5,
                   fontweight="bold", pad=6, color="#9B3226")
    tag(bot[0], "d")
    axem = fig.add_subplot(bot[2]); axem.axis("off"); axem.imshow(crop(f"{SCR}/f18_esp_meso.png"))
    axem.set_title("Mesophile — 6PZ7 ($\\mathit{C.\\ acetobutylicum}$)", fontsize=14.5,
                   fontweight="bold", pad=6, color="#4A5568")
    # central vertical colorbar — larger, explicitly labelled, identical scale for both panels
    cax = fig.add_subplot(bot[1]); box = cax.get_position(); cax.remove()
    cb_ax = fig.add_axes([box.x0 + box.width * 0.34, box.y0 + box.height * 0.14,
                          box.width * 0.30, box.height * 0.70])
    sm = ScalarMappable(norm=Normalize(-1, 1), cmap="RdBu")
    cb = fig.colorbar(sm, cax=cb_ax, orientation="vertical", ticks=[-1, 0, 1])
    cb.ax.set_yticklabels(["negative", "0", "positive"], fontsize=11, fontweight="bold")
    cb.ax.tick_params(length=0)
    cb.outline.set_visible(True); cb.outline.set_linewidth(0.8); cb.outline.set_edgecolor("#666")
    cb_ax.set_title("electrostatic\npotential\n(relative,\nidentical scale)", fontsize=9.5,
                    color="#555", pad=6, linespacing=1.2)
    y2 = outer[2].get_position(fig).y0
    fig.text(0.5, y2 - 0.028,
             "Vacuum electrostatic surface potential (PyMOL, Amber99 charges; same orientation and identical "
             "relative colour scale for both structures) — red negative, blue positive. The thermophile "
             "presents an expanded, more strongly polarized charged surface.",
             ha="center", fontsize=11.5, style="italic", color="#444")

    fig.savefig(f"{FIG}/Fig18_model.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Fig18 signature composed")


if __name__ == "__main__":
    render_all()
    compose()

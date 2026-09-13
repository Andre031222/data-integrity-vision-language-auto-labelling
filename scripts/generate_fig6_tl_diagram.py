# -*- coding: utf-8 -*-
"""Fig 6 -- Two-Stage Transfer Learning diagram (clean, no overlap)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "paper" / "figures"
DPI = 300

# Colours
S1_BG="#F3E5F5"; S1_BD="#7B1FA2"
MB_BG="#DCEEFB"; MB_BD="#1565C0"
S2_BG="#FFEBEE"; S2_BD="#C62828"
OT_BG="#E8F5E9"; OT_BD="#2E7D32"
IN_BG="#F5F5F5"; IN_BD="#546E7A"
FL_BG="#ECEFF1"; FL_BD="#455A64"
BN_BG="#FFFDE7"; BN_BD="#F57F17"
WH = "white"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

W, H = 20.0, 13.0
fig = plt.figure(figsize=(W, H))
ax  = fig.add_axes([0,0,1,1])
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.axis("off")
ax.set_facecolor(WH); fig.patch.set_facecolor(WH)

# -- helpers ------------------------------------------------------------------
def rb(x,y,w,h, fc,ec, lw=2.0, r=0.22, z=2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        fc=fc, ec=ec, lw=lw, zorder=z))

def t(x,y,s, fs=8, fw="normal", fc="black",
      ha="center", va="center", z=5, it=False):
    kw = dict(fontsize=fs, fontweight=fw, color=fc,
              ha=ha, va=va, zorder=z, multialignment=ha)
    if it: kw["style"]="italic"
    ax.text(x,y,s,**kw)

def ar(x1,y1,x2,y2, c="#455A64", lw=2.2, hw=0.20, hl=0.25):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
        arrowprops=dict(
            arrowstyle=f"-|>,head_width={hw},head_length={hl}",
            color=c, lw=lw), zorder=8)

def badge(x,y, n, bg, ec):
    ax.add_patch(plt.Circle((x,y),0.34, fc=bg, ec=ec, lw=2.2, zorder=9))
    ax.text(x,y,str(n), fontsize=10, fontweight="bold", color=ec,
            ha="center", va="center", zorder=10)

def hline(x1,x2,y, c="#cccccc", lw=0.8):
    ax.plot([x1,x2],[y,y], color=c, lw=lw, zorder=4)

# -- layout constants ----------------------------------------------------------
BOT=3.20; TOP=11.00; BH=TOP-BOT          # main boxes: y 3.2-11.0  (h=7.8)
# x positions
X_IN  = 0.25;  W_IN  = 2.30
X_S1  = 3.00;  W_S1  = 3.30
X_MB  = 6.80;  W_MB  = 4.20
X_S2  = 11.50; W_S2  = 3.30
X_OT  = 15.30; W_OT  = 2.55
# arrow centres
AY = BOT + BH*0.5   # vertical centre of main boxes

# TITLE
t(W/2, 12.62,
  "Two-Stage Transfer Learning Pipeline  --  EfficientNet-B2",
  fs=16, fw="bold", fc="#1A237E")
t(W/2, 12.20,
  "From general pre-training to alpaca-specific ocular anomaly classification",
  fs=9.5, fc="#5C6BC0")

# 1. INPUT
rb(X_IN, BOT, W_IN, BH, IN_BG, IN_BD, lw=1.8)
badge(X_IN+0.42, TOP-0.50, "1", IN_BG, IN_BD)
t(X_IN+W_IN/2, TOP-1.10, "INPUT",       fs=12, fw="bold", fc=IN_BD)
t(X_IN+W_IN/2, TOP-1.60, "ImageNet",    fs=9,  fc="#37474F")
t(X_IN+W_IN/2, TOP-2.00, "1.28M images",fs=8,  fc="#607D8B")
t(X_IN+W_IN/2, TOP-2.38, "1,000 classes",fs=8, fc="#607D8B")
# stacked card icons
for i in range(3):
    di=i*0.14
    rb(X_IN+0.30+di, BOT+0.50+di, 1.65, 2.10,
       "#CFD8DC","#90A4AE", lw=0.8, r=0.08, z=3-i)
# mountain in top card
for pts,fc_ in [
    ([[X_IN+0.40,BOT+0.90],[X_IN+0.88,BOT+1.60],[X_IN+1.36,BOT+0.90]], "#B0BEC5"),
    ([[X_IN+0.80,BOT+0.90],[X_IN+1.28,BOT+1.48],[X_IN+1.76,BOT+0.90]], "#90A4AE")]:
    ax.add_patch(plt.Polygon(pts, closed=True, fc=fc_, ec="none", zorder=4))
ax.add_patch(plt.Circle((X_IN+1.72, BOT+1.60), 0.14, fc="#FFD54F", ec="none", zorder=5))

ar(X_IN+W_IN, AY, X_S1-0.08, AY)

# 2. STAGE 1
rb(X_S1, BOT, W_S1, BH, S1_BG, S1_BD, lw=2.5)
CX1 = X_S1 + W_S1/2   # centre x of Stage 1

badge(X_S1+0.46, TOP-0.50, "2", S1_BG, S1_BD)
t(CX1, TOP-1.05, "STAGE 1",             fs=12, fw="bold", fc=S1_BD)
t(CX1, TOP-1.52, "Fundus Pre-training", fs=9.5, fw="bold", fc=S1_BD)
t(CX1, TOP-1.90, "(General Transfer)",  fs=8.5, fc="#6A1B9A", it=True)
hline(X_S1+0.20, X_S1+W_S1-0.20, TOP-2.15)

# brain icon -- y centre at TOP-3.30
BX, BY = CX1, TOP-3.30
ax.add_patch(plt.Circle((BX,BY), 0.62, fc=S1_BG, ec=S1_BD, lw=2.0, zorder=5))
for dx,dy,r in [(-0.20,0.16,0.22),(0.20,0.16,0.22),
                (-0.24,-0.12,0.18),(0.24,-0.12,0.18),(0,-0.26,0.20)]:
    ax.add_patch(plt.Circle((BX+dx,BY+dy),r, fc="#CE93D8", ec=S1_BD,
                             lw=1.0, zorder=6))

# description text BELOW icon
t(CX1, BY-0.90, "Trained on fundus images\nto learn general eye features.\n"
                "Backbone frozen; only\nclassification head updated.",
  fs=7.8, fc="#4A148C")

# settings box at bottom
SB1Y = BOT+0.15; SB1H = 2.00
rb(X_S1+0.18, SB1Y, W_S1-0.36, SB1H, "#EDE7F6", S1_BD, lw=1.2, r=0.15)
t(CX1, SB1Y+SB1H-0.25, "Typical settings", fs=8, fw="bold", fc=S1_BD)
hline(X_S1+0.28, X_S1+W_S1-0.28, SB1Y+SB1H-0.45, c=S1_BD, lw=0.6)
for i, p in enumerate(["LR: 1e-3  (head only)",
                        "Epochs: 30  |  Batch: 32",
                        "Optimizer: AdamW",
                        "Data: 1,896 fundus images"]):
    t(CX1, SB1Y+SB1H-0.80-i*0.30, p, fs=7.5, fc="#4A148C")

ar(X_S1+W_S1, AY, X_MB-0.08, AY, c=S1_BD)
t((X_S1+W_S1+X_MB)/2, AY+0.45, "transfer\nweights", fs=7.5, fc=S1_BD)

# 3. EFFICIENTNET-B2
rb(X_MB, BOT, W_MB, BH, MB_BG, MB_BD, lw=2.5)
CMX = X_MB + W_MB/2

badge(X_MB+0.46, TOP-0.50, "3", MB_BG, MB_BD)
t(CMX, TOP-1.05, "BASE MODEL",      fs=12, fw="bold", fc=MB_BD)
t(CMX, TOP-1.52, "EfficientNet-B2", fs=9.5, fw="bold", fc=MB_BD)
t(CMX, TOP-1.90, "(Pre-trained backbone, 2.58M params)", fs=8, fc="#1565C0", it=True)
hline(X_MB+0.20, X_MB+W_MB-0.20, TOP-2.12)

# layer blocks -- 8 blocks in height BH - 2.20 - 0.20 = 5.40 -> each block 5.40/8 = 0.675
N_LAYERS = 8
BLK_H = (BH - 2.35) / N_LAYERS   # ≈ 0.68
layers = [
    ("Stem",    "Conv 3x3, stride 2",    "#BBDEFB", "#0D47A1"),
    ("Block 1", "MBConv1,  k3x3",        "#90CAF9", "#0D47A1"),
    ("Block 2", "MBConv6,  k3x3",        "#64B5F6", "#0D47A1"),
    ("Block 3", "MBConv6,  k5x5",        "#42A5F5", WH),
    ("Block 4", "MBConv6,  k5x5",        "#2196F3", WH),
    ("Block 5", "MBConv6,  k3x3",        "#1E88E5", WH),
    ("Block 6", "MBConv6,  k3x3",        "#1976D2", WH),
    ("Head+GAP","Pooling + FC (binary)", "#1565C0", WH),
]
for i,(name,detail,bg,tc) in enumerate(layers):
    ly = BOT + 0.18 + (N_LAYERS-1-i)*BLK_H
    rb(X_MB+0.18, ly, W_MB-0.36, BLK_H-0.06, bg, MB_BD, lw=0.9, r=0.08, z=4)
    # small stack icon
    for ki in range(3):
        rb(X_MB+0.26+ki*0.06, ly+0.06+ki*0.05, 0.36, BLK_H-0.22,
           "#E3F2FD", MB_BD, lw=0.5, r=0.04, z=5)
    tx0 = X_MB + 0.80
    ax.text(tx0, ly+BLK_H*0.68, name,   fontsize=8.5, fontweight="bold",
            color=tc, ha="left", va="center", zorder=6)
    ax.text(tx0, ly+BLK_H*0.25, detail, fontsize=7.5,
            color=tc, ha="left", va="center", zorder=6)

ar(X_MB+W_MB, AY, X_S2-0.08, AY, c=S2_BD)

# 4. STAGE 2
rb(X_S2, BOT, W_S2, BH, S2_BG, S2_BD, lw=2.5)
CX2 = X_S2 + W_S2/2

badge(X_S2+0.46, TOP-0.50, "4", S2_BG, S2_BD)
t(CX2, TOP-1.05, "STAGE 2",            fs=12, fw="bold", fc=S2_BD)
t(CX2, TOP-1.52, "Alpaca Fine-tuning", fs=9.5, fw="bold", fc=S2_BD)
t(CX2, TOP-1.90, "(Specialisation)",   fs=8.5, fc="#B71C1C", it=True)
hline(X_S2+0.20, X_S2+W_S2-0.20, TOP-2.15)

# target icon -- y centre at TOP-3.30
TX, TY = CX2, TOP-3.30
for r_,c_ in [(0.58,S2_BG),(0.44,S2_BD),(0.30,S2_BG),(0.16,S2_BD)]:
    ax.add_patch(plt.Circle((TX,TY), r_, fc=c_, ec=S2_BD, lw=1.2, zorder=5))
ax.annotate("", xy=(TX+0.06, TY),
            xytext=(TX+0.70, TY+0.42),
            arrowprops=dict(arrowstyle="-|>", color="#FF6F00",
                            lw=2.8, mutation_scale=14), zorder=7)

t(CX2, TY-0.92, "Fine-tuned on alpaca eye\ncrops to specialise for\n"
                 "anomaly detection.\nFull network updated.",
  fs=7.8, fc="#7F0000")

# settings box
SB2Y = BOT+0.15; SB2H = 2.20
rb(X_S2+0.18, SB2Y, W_S2-0.36, SB2H, "#FFCDD2", S2_BD, lw=1.2, r=0.15)
t(CX2, SB2Y+SB2H-0.25, "Typical settings", fs=8, fw="bold", fc=S2_BD)
hline(X_S2+0.28, X_S2+W_S2-0.28, SB2Y+SB2H-0.45, c=S2_BD, lw=0.6)
for i, p in enumerate(["LR: 1e-4  (full fine-tune)",
                        "Epochs: 60  |  Batch: 32",
                        "Optimizer: AdamW",
                        "Focal Loss: gamma=1.5, alpha=balanced",
                        "Data: 462 real crops (91 anom / 371 norm)"]):
    t(CX2, SB2Y+SB2H-0.80-i*0.30, p, fs=7.5, fc="#7F0000")

ar(X_S2+W_S2, AY, X_OT-0.08, AY, c=OT_BD)

# 5. OUTPUT
rb(X_OT, BOT, W_OT, BH, OT_BG, OT_BD, lw=2.5)
COX = X_OT + W_OT/2

badge(X_OT+0.42, TOP-0.50, "5", OT_BG, OT_BD)
t(COX, TOP-1.05, "OUTPUT",      fs=12, fw="bold", fc=OT_BD)
t(COX, TOP-1.52, "Predictions", fs=9.5, fw="bold", fc=OT_BD)
hline(X_OT+0.20, X_OT+W_OT-0.20, TOP-1.80)

# bar chart (chance-level: AUC ~ 0.5)
bar_vals  = [0.51, 0.50, 0.49]
bar_cols  = [S2_BD, OT_BD, S1_BD]
bar_bot   = BOT + 4.20
bar_maxh  = 2.00
bar_w     = 0.48
bar_gap   = 0.15
bx0 = COX - (3*bar_w + 2*bar_gap)/2
for i,(h,c) in enumerate(zip(bar_vals, bar_cols)):
    bxi = bx0 + i*(bar_w+bar_gap)
    rb(bxi, bar_bot, bar_w, h*bar_maxh, c, c, lw=0, r=0.06, z=5)
ax.plot([bx0-0.1, bx0+3*bar_w+2*bar_gap+0.1],
        [bar_bot, bar_bot], color=OT_BD, lw=1.0, zorder=6)

t(COX, bar_bot+bar_maxh+0.35, "Honest test metrics", fs=8, fw="bold", fc=OT_BD)
t(COX, BOT+3.70, "AUC-ROC = 0.506", fs=8.5, fw="bold", fc=OT_BD)
t(COX, BOT+3.30, "F1        = 0.065", fs=8.5, fw="bold", fc=OT_BD)
t(COX, BOT+2.90, "Acc = 0.586  (chance)", fs=8.0, fc="#B71C1C")
hline(X_OT+0.20, X_OT+W_OT-0.20, BOT+2.65, c=OT_BD)
t(COX, BOT+2.10, "Binary output:", fs=8, fc="#37474F")
t(COX, BOT+1.65, "Anomaly", fs=9, fw="bold", fc=S2_BD)
t(COX, BOT+1.28, "Normal",  fs=9, fw="bold", fc=MB_BD)

# FLOW BAR
FB_Y=2.36; FB_H=0.82
rb(0.22, FB_Y, W-0.44, FB_H, FL_BG, FL_BD, lw=1.2, r=0.15)
t(1.20, FB_Y+FB_H/2, "GENERAL\nFLOW", fs=7.5, fw="bold", fc=FL_BD)
flow_items = [
    (3.80, "General dataset (ImageNet)"),
    (6.55, "Stage 1 pre-training"),
    (9.45, "Pre-trained model weights"),
    (12.35,"Stage 2 fine-tuning"),
    (15.60,"Final model for inference"),
]
for i,(fx,label) in enumerate(flow_items):
    rb(fx-0.24, FB_Y+0.29, 0.48, 0.44, FL_BD, FL_BD, lw=0, r=0.10, z=6)
    t(fx, FB_Y+0.51, str(i+1), fs=9, fw="bold", fc=WH, z=7)
    t(fx, FB_Y+0.13, label, fs=7.2, fc=FL_BD, z=6)
    if i < len(flow_items)-1:
        ar(fx+0.56, FB_Y+FB_H/2, fx+1.00, FB_Y+FB_H/2,
           c=FL_BD, lw=1.4, hw=0.09, hl=0.11)

# BENEFITS
BF_Y=0.18; BF_H=2.05
rb(0.22, BF_Y, 12.00, BF_H, BN_BG, BN_BD, lw=1.5, r=0.15)
t(0.75, BF_Y+BF_H-0.26, "BENEFITS OF THE TWO-STAGE APPROACH",
  fs=8.5, fw="bold", fc=BN_BD, ha="left")
hline(0.40, 12.05, BF_Y+BF_H-0.46, c=BN_BD, lw=0.6)
benefits = [
    "Leverages prior\nknowledge (TL)",
    "Reduces data\nrequirements",
    "Improves perf. on\nsmall datasets",
    "Better generalisation\nand robustness",
    "Easily adaptable\nto new domains",
]
for i,txt_ in enumerate(benefits):
    bx2 = 1.10 + i*2.38
    ax.add_patch(plt.Circle((bx2, BF_Y+1.30), 0.26, fc=OT_BD, ec="none", zorder=6))
    t(bx2, BF_Y+1.30, "OK", fs=7.5, fw="bold", fc=WH, z=7)
    t(bx2, BF_Y+0.58, txt_, fs=7.5, fc="#5D4037", z=6)

# LEGEND
LG_X=12.48; LG_Y=0.18; LG_W=7.30; LG_H=2.05
rb(LG_X, LG_Y, LG_W, LG_H, "#FAFAFA", "#90A4AE", lw=1.2, r=0.15)
t(LG_X+0.30, LG_Y+LG_H-0.26, "LEGEND",
  fs=8.5, fw="bold", fc="#37474F", ha="left")
hline(LG_X+0.20, LG_X+LG_W-0.20, LG_Y+LG_H-0.46, c="#90A4AE", lw=0.6)
for i,(c_,label) in enumerate([
    (S1_BD, "Stage 1: Fundus pre-training (general transfer)"),
    (MB_BD, "Base model: EfficientNet-B2 pre-trained backbone"),
    (S2_BD, "Stage 2: Alpaca fine-tuning (specialisation)"),
    (OT_BD, "Output: Adapted model ready for inference"),
]):
    ly2 = LG_Y+LG_H-0.90 - i*0.38
    rb(LG_X+0.28, ly2-0.12, 0.30, 0.26, c_, c_, lw=0, r=0.05, z=6)
    t(LG_X+0.72, ly2, label, fs=7.8, fc="#37474F", ha="left", z=6)

# FOOTNOTE
t(W/2, 0.07,
  "Stage 1 dataset: RFMiD-2.0 (human fundus, CC BY 4.0).   "
  "Stage 2 dataset: auto-labelled alpaca ocular crops (Groq Vision + manual review).",
  fs=7.2, fc="#90A4AE")

for ext in ("png","pdf"):
    fig.savefig(OUT/f"fig6_transfer_learning.{ext}",
                dpi=DPI, bbox_inches="tight", facecolor=WH)
plt.close(fig)
print("Saved fig6_transfer_learning.png / .pdf")

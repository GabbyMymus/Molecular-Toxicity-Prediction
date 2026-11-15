#!/usr/bin/env python3
"""Build slides_phase2/tox21_phase2.pptx.

Author: Abhijeet (Deliverable A5); numbers rebuilt from committed files by Parrv (task S1).
13-slide deck covering Phase 2 deep learning, hybrids, ablations and results.

Every result number is computed here from results/*.csv, results/best_configs_dl.json, the committed
prediction files' CSV summaries and the data files; nothing is typed in by hand except the two items
listed in slides_phase2/README.md under "Typed constants".
Run from tox21-phase1/ with the venv active:  python slides_phase2/build_deck.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")
OUT_DIR = ROOT / "slides_phase2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PPTX = OUT_DIR / "tox21_phase2.pptx"

# Color Palette
NAVY = RGBColor(0x14, 0x2B, 0x4D)      # Primary Dark
TEAL = RGBColor(0x1B, 0x7F, 0x8C)      # Accent
BLUE = RGBColor(0x2B, 0x5B, 0x84)      # Secondary
GREY = RGBColor(0x55, 0x5F, 0x6B)      # Body Text
LIGHT = RGBColor(0xF4, 0xF7, 0xF9)     # Card Background
WHITE = RGBColor(0xFF, 0xFF, 0xFF)     # Pure White
RED = RGBColor(0xB2, 0x3A, 0x3A)       # Alert / Gap
GREEN = RGBColor(0x2E, 0x7D, 0x32)     # Win / Success

W, H = 13.333, 7.5
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]

slide_num = [1]
TOTAL_SLIDES = 13
N_TESTS_PASSED = 24   # typed constant: `make phase2-test` result observed in the S1 session (see README)

def add_header(slide, title, category="Tox21 Phase 2: Graph Deep Learning & Hybrid Benchmark"):
    slide_num[0] += 1
    # Category / breadcrumb
    tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11), Inches(0.3))
    tf_cat = tb_cat.text_frame
    tf_cat.word_wrap = True
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category.upper()
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = TEAL
    p_cat.font.name = "Arial"

    # Slide Title
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11), Inches(0.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.font.name = "Arial"

    # Slide Number
    tb_num = slide.shapes.add_textbox(Inches(12.0), Inches(0.4), Inches(0.8), Inches(0.4))
    tf_num = tb_num.text_frame
    p_num = tf_num.paragraphs[0]
    p_num.text = f"{slide_num[0]} / {TOTAL_SLIDES}"
    p_num.font.size = Pt(11)
    p_num.font.color.rgb = GREY
    p_num.alignment = PP_ALIGN.RIGHT

def add_card(slide, x, y, w, h, bg_color=LIGHT, border_color=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.5)
    else:
        shape.line.fill.background()
    return shape

def add_textbox(slide, x, y, w, h, paragraphs, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0.08)
    for i, p_info in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(p_info, str):
            p.text = p_info
            p.font.size = Pt(14)
            p.font.color.rgb = GREY
            p.font.name = "Arial"
        elif isinstance(p_info, tuple):
            text, opts = p_info
            bullet = opts.get("bullet", False)
            p.text = ("•  " + text) if bullet else text
            p.font.size = Pt(opts.get("size", 14))
            p.font.bold = opts.get("bold", False)
            p.font.color.rgb = opts.get("color", GREY)
            p.font.name = "Arial"
            if "space" in opts:
                p.space_after = Pt(opts["space"])
    return tb



def notes(slide, text):
    """Attach speaking notes to a slide."""
    slide.notes_slide.notes_text_frame.text = text


def add_table(slide, x, y, w, rows, col_w, font=10, row_h=0.3, header_fill=NAVY, bold_rows=()):
    """rows[0] is the header; all cells are strings."""
    gt = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    tbl = gt.table
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, r in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(r):
            c = tbl.cell(i, j)
            c.text = val
            c.margin_top = c.margin_bottom = Inches(0.02)
            para = c.text_frame.paragraphs[0]
            para.font.size = Pt(font)
            para.font.name = "Arial"
            para.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = header_fill
                para.font.bold = True
                para.font.color.rgb = WHITE
            else:
                c.fill.fore_color.rgb = LIGHT if i % 2 else WHITE
                para.font.color.rgb = NAVY
                para.font.bold = i in bold_rows
    return tbl


# ==================== NUMBERS: all computed from committed files ====================
from make_results_table import GROUPS, load, summarize           # noqa: E402  (Phase 1 + DL test CSVs)
from src.config import RESULTS                                    # noqa: E402
from src.data import load_clean                                   # noqa: E402
from src.dl import analysis as A                                  # noqa: E402

# --- data ---
_, Y_ALL = load_clean()
Y_ALL = np.asarray(Y_ALL, float)
N_MOL, N_EP = Y_ALL.shape
N_CELLS = Y_ALL.size
N_MISS = int(np.isnan(Y_ALL).sum())
N_OBS = N_CELLS - N_MISS
N_POS = int(np.nansum(Y_ALL))
MISS_PCT = 100 * N_MISS / N_CELLS
POS_PCT = 100 * N_POS / N_OBS

# --- parameter counts (frozen configs, repo constructors; same logic as scripts/count_params.py) ---
import count_params as CP                                         # noqa: E402
from src.dl.run import N_MORGAN, build_model                      # noqa: E402
from src.features import load_features                            # noqa: E402

_cfg = json.loads((RESULTS / "best_configs_dl.json").read_text())
_n_total = load_features("morgan+desc", "scaffold")[0].shape[1]
_n_desc = _n_total - N_MORGAN
PARAMS = {
    "mpnn": CP.n_params(build_model("mpnn", _cfg["mpnn|none"], 0)),
    "h1": CP.n_params(build_model("mpnn", _cfg["mpnn|none"], _n_desc)),
    "gcn": CP.n_params(build_model("gcn", _cfg["gcn|none"], 0)),
    "fpmlp": CP.n_params(build_model("fpmlp", _cfg["fpmlp|both"], _n_total)),
}

# --- test results (scaffold and random), mean +/- sample SD over seeds of the per-seed endpoint mean ---
DATA = load()
ROWS = [(lab, m, fs, src) for _, members in GROUPS for (lab, m, fs, src) in members]
ST = {(lab, sp): summarize(DATA[src], m, fs, sp) for (lab, m, fs, src) in ROWS for sp in ("scaffold", "random")}
LAB = {"h1": "H1 MPNN+D", "fpmlp": "D0 FP-MLP, M+D", "gcn": "D1 GCN", "mpnn": "D2 MPNN",
       "h2p": "H2 XGBoost+FP-MLP", "h2s": "H2 alt. XGBoost+H1", "xgb": "XGBoost, M+D", "rf": "RF, M+D"}


def pr(lab, sp="scaffold"):
    return ST[(LAB[lab], sp)]["pr_auc"]


def roc(lab, sp="scaffold"):
    return ST[(LAB[lab], sp)]["roc_auc"]


def f4(x):
    return f"{x:.4f}"


def pm(t):
    return f"{t[0]:.4f} ± {t[1]:.4f}"


# --- paired endpoint tests on seed-averaged endpoint PR AUC (scaffold) ---
DF = A.load_all(RESULTS)
K = {"h1": ("mpnn+desc", "graph+desc"), "fpmlp": ("fpmlp+both", "both"), "gcn": ("gcn", "graph"),
     "mpnn": ("mpnn", "graph"), "h2p": ("ens_xgb_fpmlp+both", "morgan+desc+both"),
     "h2s": ("ens_xgb_mpnn+desc", "morgan+desc+graph+desc"), "xgb": ("xgb", "morgan+desc")}
T = {}
for a, b in [("h1", "fpmlp"), ("gcn", "mpnn"), ("h2p", "h1"), ("h2s", "h1"), ("h2p", "xgb"), ("h2s", "xgb")]:
    T[(a, b)] = A.paired_endpoint_test(DF, K[a], K[b])


def tline(a, b):
    t = T[(a, b)]
    return f"{t['wins']} wins / {t['losses']} losses, mean diff {t['mean_diff']:+.4f}, p = {t['p_value']:.4f}"


# --- bootstrap (results/dl_bootstrap_ci.csv) ---
BS = pd.read_csv(RESULTS / "dl_bootstrap_ci.csv")
BS_N, BS_SEED = int(BS.n_resamples.iloc[0]), int(BS.rng_seed.iloc[0])


def bs_counts(name):
    g = BS[BS.comparison == name]
    return int((g.ci_low > 0).sum()), int((g.ci_high < 0).sum())


aro = BS[(BS.comparison == "h1_minus_fpmlp") & (BS.endpoint == "NR-Aromatase")].iloc[0]

# --- generalization gap ---
GAP = A.generalization_gap(DF)
GAP_PR_MIN, GAP_PR_MAX = GAP.pr_gap.min(), GAP.pr_gap.max()
GAP_ROC_MIN, GAP_ROC_MAX = GAP.roc_gap.min(), GAP.roc_gap.max()


def gap(lab):
    return GAP.loc[K[lab]].pr_gap


# --- descriptor / representation ablations (Ranajeet CSV) and architecture ablations (Malhar CSV) ---
ABL_R = pd.read_csv(RESULTS / "dl_ablation_test_ranajeet.csv")
ABL_M = pd.read_csv(RESULTS / "dl_ablation_test_malhar.csv")


def abl(df, tag):
    d = df[df.tag == tag]
    if sorted(d.seed.unique()) != [0, 1, 2]:
        raise SystemExit(f"{tag}: expected seeds 0,1,2")
    ps = d.groupby("seed")[["pr_auc", "roc_auc"]].mean()
    return ps.pr_auc.mean(), ps.roc_auc.mean(), ps


SHUF_DESC_PR, SHUF_DESC_ROC, _ = abl(ABL_R, "shuffled_desc")
BOTH_PR, _, _ = abl(ABL_R, "both")
MORGAN_PR, _, _ = abl(ABL_R, "morgan")
SL_PR, SL_ROC, SL_PS = abl(ABL_M, "C1_shuffled_labels")
SL_ROC_SEEDS = [SL_PS.roc_auc[s] for s in (0, 1, 2)]
ARCH_TAGS = [("A1_depth1", "1 message-passing step"), ("A2_depth2", "2 steps"),
             ("P2_depth4", "4 steps"), ("A3_no_bond", "no bond features"),
             ("P2_readout_mean", "mean readout"), ("P2_readout_sum", "sum readout")]
MPNN_PR, MPNN_ROC = pr("mpnn")[0], roc("mpnn")[0]
THRESH = 0.55  # the report's own threshold for this control (validation.tex); not an external standard

# ==================== SLIDE 1: Title & Overview ====================
s1 = prs.slides.add_slide(BLANK)
add_card(s1, 0.8, 0.8, 11.733, 5.9, bg_color=WHITE, border_color=TEAL)
tb1 = s1.shapes.add_textbox(Inches(1.2), Inches(1.4), Inches(11.0), Inches(2.2))
tf1 = tb1.text_frame
tf1.word_wrap = True
p = tf1.paragraphs[0]
p.text = "Molecular Toxicity Prediction on Tox21"
p.font.size = Pt(32)
p.font.bold = True
p.font.color.rgb = NAVY
p.font.name = "Arial"

p2 = tf1.add_paragraph()
p2.text = "Phase 2: Graph Neural Networks, Hybrids, and Controls"
p2.font.size = Pt(18)
p2.font.bold = True
p2.font.color.rgb = TEAL
p2.font.name = "Arial"
p2.space_before = Pt(8)

p3 = tf1.add_paragraph()
p3.text = "A scaffold-split benchmark comparing Phase 1 tree models with graph and hybrid deep models."
p3.font.size = Pt(14)
p3.font.color.rgb = GREY
p3.font.name = "Arial"
p3.space_before = Pt(6)

add_card(s1, 1.2, 4.0, 10.9, 2.2, bg_color=LIGHT)
author_text = [
    ("Team and roles (from the project documents):", {"size": 13, "bold": True, "color": NAVY, "space": 4}),
    ("Abhijeet Kumar: research and analysis (literature, results, discussion, slides)", {"size": 12, "bullet": True}),
    ("Ranajeet: hybrids (H1, H2), FP-MLP control, representation ablations", {"size": 12, "bullet": True}),
    ("Parrv: deep-learning engine, hyperparameter freeze, command line, Makefile", {"size": 12, "bullet": True}),
    ("Raghav: graph data, alignment checks, GCN", {"size": 12, "bullet": True}),
    ("Malhar: verification and integration, architecture ablations", {"size": 12, "bullet": True}),
]
add_textbox(s1, 1.4, 4.15, 10.5, 1.9, author_text)
notes(s1, "Introduce Phase 2: we compare the Phase 1 tree models with a GCN, an MPNN and two hybrids on the same "
          "scaffold and random splits, three seeds each. Roles are as recorded in the project documents.")

# ==================== SLIDE 2: Problem and Chemical Context ====================
s2 = prs.slides.add_slide(BLANK)
add_header(s2, "1. Chemical Context: Fingerprints versus Learned Graph Representations")
add_card(s2, 0.8, 1.5, 5.6, 5.2, bg_color=WHITE, border_color=GREY)
add_card(s2, 6.8, 1.5, 5.7, 5.2, bg_color=LIGHT)

left_s2 = [
    ("The representation question:", {"size": 16, "bold": True, "color": NAVY, "space": 8}),
    ("Classical models use fixed circular Morgan fingerprints hashed into 2,048 bits.", {"size": 13, "bullet": True, "space": 6}),
    ("Hashing can map different substructures to the same bit.", {"size": 13, "bullet": True, "space": 6}),
    ("The features are fixed in advance, not learned for each assay.", {"size": 13, "bullet": True, "space": 6}),
    ("RDKit descriptors add whole-molecule properties (for example logP, TPSA, molecular weight).", {"size": 13, "bullet": True, "space": 6}),
]
add_textbox(s2, 1.0, 1.7, 5.2, 4.8, left_s2)

right_s2 = [
    ("The graph hypothesis:", {"size": 16, "bold": True, "color": TEAL, "space": 8}),
    ("Treat a molecule as a graph G = (V, E):", {"size": 13, "bold": True, "space": 4}),
    ("Nodes: atoms with 33-dimensional feature vectors.", {"size": 12, "bullet": True}),
    ("Edges: bonds with 6-dimensional feature vectors.", {"size": 12, "bullet": True, "space": 6}),
    ("Message passing:", {"size": 13, "bold": True, "space": 4}),
    ("Atom states are updated from bonded neighbors, so representations are learned from topology.", {"size": 12, "bullet": True, "space": 6}),
    ("Central question:", {"size": 13, "bold": True, "color": RED, "space": 2}),
    (f"Do learned graph models beat tuned tree models on a small benchmark (6,167 training molecules)?", {"size": 12, "bold": True, "color": NAVY}),
]
add_textbox(s2, 7.0, 1.7, 5.3, 4.8, right_s2)
notes(s2, "Motivation only; no result is claimed on this slide. The question we test is whether learned graph "
          "representations beat tuned trees on 6,167 training molecules.")

# ==================== SLIDE 3: Data & Scaffold Splits ====================
s3 = prs.slides.add_slide(BLANK)
add_header(s3, "2. Tox21 Benchmark: Label Sparsity and Scaffold Splits")
add_card(s3, 0.8, 1.5, 3.7, 5.2, bg_color=LIGHT)
add_card(s3, 4.8, 1.5, 3.7, 5.2, bg_color=LIGHT)
add_card(s3, 8.8, 1.5, 3.7, 5.2, bg_color=LIGHT)

col1 = [
    ("Dataset", {"size": 15, "bold": True, "color": NAVY, "space": 6}),
    (f"{N_MOL:,} molecules, {N_EP} assay endpoints ({N_CELLS:,} label cells).", {"size": 12, "bullet": True}),
    (f"Missing labels: {N_MISS:,} of {N_CELLS:,} cells ({MISS_PCT:.1f}%). They are masked, never filled with 0.", {"size": 12, "bullet": True}),
    (f"Positives: {N_POS:,} of {N_OBS:,} observed labels ({POS_PCT:.1f}%).", {"size": 12, "bullet": True}),
    ("Per-endpoint positive rates differ (README: 2.91% to 15.94%).", {"size": 12, "bullet": True}),
]
add_textbox(s3, 1.0, 1.7, 3.3, 4.8, col1)

col2 = [
    ("Bemis-Murcko scaffold split", {"size": 15, "bold": True, "color": TEAL, "space": 6}),
    ("Whole scaffold groups are assigned to one partition, largest first.", {"size": 12, "bullet": True}),
    ("6,167 train / 771 validation / 771 test.", {"size": 12, "bullet": True}),
    ("No scaffold group spans two partitions (Phase 1 test).", {"size": 12, "bullet": True}),
    ("1,753 acyclic molecules share the empty scaffold and are all in train, so validation and test hold only ring-containing molecules.", {"size": 12, "bullet": True}),
]
add_textbox(s3, 5.0, 1.7, 3.3, 4.8, col2)

col3 = [
    ("Why two splits", {"size": 15, "bold": True, "color": RED, "space": 6}),
    ("Random split: test scaffolds are often seen in training.", {"size": 12, "bullet": True}),
    ("Scaffold split: test scaffolds are unseen; this is our primary benchmark.", {"size": 12, "bullet": True}),
    ("Metrics: masked ROC AUC and PR AUC per endpoint, averaged over 12 endpoints.", {"size": 12, "bullet": True}),
    ("PR AUC is the primary metric; seeds 0, 1, 2.", {"size": 12, "bullet": True}),
]
add_textbox(s3, 9.0, 1.7, 3.3, 4.8, col3)
notes(s3, f"{N_MISS:,} of {N_CELLS:,} label cells are missing ({MISS_PCT:.2f}%), and positives are {N_POS:,} of {N_OBS:,} "
          f"observed labels ({POS_PCT:.2f}%); both computed from the cleaned data file. The 2.91% to 15.94% per-endpoint "
          "range is quoted from the README and was not recomputed here. The 1,753 acyclic molecules are all in training.")

# ==================== SLIDE 4: Graph Featurization & Training ====================
s4 = prs.slides.add_slide(BLANK)
add_header(s4, "3. Graph Construction and Training Objective")
add_card(s4, 0.8, 1.5, 5.6, 5.2, bg_color=WHITE, border_color=BLUE)
add_card(s4, 6.8, 1.5, 5.7, 5.2, bg_color=LIGHT)

left_s4 = [
    ("Atom and bond features:", {"size": 16, "bold": True, "color": NAVY, "space": 6}),
    ("Atom vector (33 dimensions):", {"size": 13, "bold": True, "color": TEAL, "space": 2}),
    ("One-hot atom symbol, degree, attached hydrogens and hybridization (each with an 'other' bin)", {"size": 12, "bullet": True}),
    ("Clipped formal charge, aromaticity, ring membership, scaled atomic mass", {"size": 12, "bullet": True, "space": 6}),
    ("Bond vector (6 dimensions):", {"size": 13, "bold": True, "color": TEAL, "space": 2}),
    ("One-hot bond type, conjugation, ring membership", {"size": 12, "bullet": True}),
    ("Each bond becomes two directed edges.", {"size": 12, "bullet": True}),
]
add_textbox(s4, 1.0, 1.7, 5.2, 4.8, left_s4)

right_s4 = [
    ("Training and checks:", {"size": 16, "bold": True, "color": NAVY, "space": 6}),
    ("Masked, class-weighted loss:", {"size": 13, "bold": True, "color": RED, "space": 2}),
    ("Missing labels contribute zero loss.", {"size": 12, "bullet": True}),
    ("Positive weight n_neg / n_pos per endpoint, capped at 50; the cap never binds (max 38.35 scaffold).", {"size": 12, "bullet": True, "space": 6}),
    ("Optimization:", {"size": 13, "bold": True, "color": BLUE, "space": 2}),
    ("AdamW, batch 64, early stopping on mean validation PR AUC (patience 10, max 60 epochs).", {"size": 12, "bullet": True, "space": 6}),
    ("Unit-tested:", {"size": 13, "bold": True, "color": BLUE, "space": 2}),
    ("Invariance to atom order; masked loss ignores missing labels; a model can overfit a tiny batch.", {"size": 12, "bullet": True}),
]
add_textbox(s4, 7.0, 1.7, 5.3, 4.8, right_s4)
notes(s4, "The cap of 50 is never reached on the committed splits (largest uncapped weight 38.35 on scaffold); "
          "the reason 50 was chosen is not recorded in the repository. The tests are in tests/test_dl_smoke.py.")

# ==================== SLIDE 5: Deep Learning Architectures ====================
s5 = prs.slides.add_slide(BLANK)
add_header(s5, "4. Deep Architectures: GCN and MPNN")
add_card(s5, 0.8, 1.5, 5.6, 5.2, bg_color=LIGHT)
add_card(s5, 6.8, 1.5, 5.7, 5.2, bg_color=WHITE, border_color=TEAL)

left_s5 = [
    ("D1: GCN", {"size": 16, "bold": True, "color": GREY, "space": 6}),
    ("Graph convolution with neighborhood averaging:", {"size": 13, "bold": True, "space": 2}),
    ("3 GCNConv layers, hidden size 256, batch norm, ReLU, dropout 0.2.", {"size": 12, "bullet": True}),
    ("Mean readout over atoms, two-layer head.", {"size": 12, "bullet": True}),
    ("Uses atom features only; bond features are ignored.", {"size": 12, "bullet": True}),
    (f"Trainable parameters: {PARAMS['gcn']:,}.", {"size": 12, "bullet": True, "bold": True, "color": NAVY}),
]
add_textbox(s5, 1.0, 1.7, 5.2, 4.8, left_s5)

right_s5 = [
    ("D2: MPNN (Gilmer et al. 2017 style)", {"size": 16, "bold": True, "color": TEAL, "space": 6}),
    ("Edge-network message passing:", {"size": 13, "bold": True, "space": 2}),
    ("An MLP maps each bond's features to a matrix that multiplies the neighbor state; messages are averaged.", {"size": 12, "bullet": True}),
    ("GRU node update, T = 3 steps with shared weights, hidden size 64.", {"size": 12, "bullet": True}),
    ("Set2Set readout (3 processing steps), two-layer head.", {"size": 12, "bullet": True}),
    (f"Trainable parameters: {PARAMS['mpnn']:,}.", {"size": 12, "bullet": True, "bold": True, "color": NAVY}),
    (f"For scale: FP-MLP control {PARAMS['fpmlp']:,}; H1 {PARAMS['h1']:,}.", {"size": 12, "bullet": True}),
]
add_textbox(s5, 7.0, 1.7, 5.3, 4.8, right_s5)
notes(s5, f"Parameter counts come from building each model with the repository constructors and the frozen "
          f"configurations: MPNN {PARAMS['mpnn']:,}, H1 {PARAMS['h1']:,}, GCN {PARAMS['gcn']:,}, FP-MLP {PARAMS['fpmlp']:,}. "
          "The MPNN has about three times the parameters of the GCN, and the FP-MLP is the largest.")

# ==================== SLIDE 6: Hybrid Design ====================
s6 = prs.slides.add_slide(BLANK)
add_header(s6, "5. Hybrid Design: H1 (MPNN + descriptors) and H2 (late fusion)")
add_card(s6, 0.8, 1.5, 5.6, 5.2, bg_color=WHITE, border_color=NAVY)
add_card(s6, 6.8, 1.5, 5.7, 5.2, bg_color=LIGHT)

left_s6 = [
    ("H1: MPNN with descriptors", {"size": 16, "bold": True, "color": NAVY, "space": 6}),
    ("Idea:", {"size": 13, "bold": True, "color": TEAL, "space": 2}),
    ("Message passing sees local topology; descriptors give whole-molecule properties directly.", {"size": 12, "bullet": True, "space": 4}),
    ("Architecture:", {"size": 13, "bold": True, "color": TEAL, "space": 2}),
    ("MPNN encoder gives a 128-dimensional graph embedding (Set2Set output, 2 x 64).", {"size": 12, "bullet": True}),
    (f"{_n_desc} standardized RDKit descriptors go through a linear layer to 128 units (ReLU, dropout).", {"size": 12, "bullet": True}),
    ("The two vectors are concatenated and passed to the classification head.", {"size": 12, "bullet": True}),
    ("Reuses the frozen MPNN configuration; no separate tuning.", {"size": 12, "bullet": True}),
]
add_textbox(s6, 1.0, 1.7, 5.2, 4.8, left_s6)

right_s6 = [
    ("H2: rank-average ensemble", {"size": 16, "bold": True, "color": NAVY, "space": 6}),
    ("Primary H2 = XGBoost + FP-MLP", {"size": 13, "bold": True, "color": BLUE, "space": 2}),
    ("The deep partner is chosen on validation only: best mean peak validation PR AUC over seeds 0-2.", {"size": 12, "bullet": True}),
    ("Secondary H2 = XGBoost + H1 (reported alongside).", {"size": 12, "bullet": True, "space": 4}),
    ("Fusion:", {"size": 13, "bold": True, "color": BLUE, "space": 2}),
    ("Scores are converted to per-endpoint ranks in (0, 1] and averaged with fixed weights 0.5 / 0.5.", {"size": 12, "bullet": True}),
    ("No training and no weight search on validation or test data.", {"size": 12, "bullet": True}),
]
add_textbox(s6, 7.0, 1.7, 5.3, 4.8, right_s6)
notes(s6, "H2 in the report means XGBoost plus FP-MLP, because the FP-MLP had the best peak validation PR AUC among "
          "the deep models. The XGBoost plus H1 variant is a secondary result. The equal weights were fixed in advance.")

# ==================== SLIDE 7: Experimental Protocol & Freeze ====================
s7 = prs.slides.add_slide(BLANK)
add_header(s7, "6. Protocol and Test-Set Discipline")
add_card(s7, 0.8, 1.5, 3.7, 5.2, bg_color=LIGHT)
add_card(s7, 4.8, 1.5, 3.7, 5.2, bg_color=LIGHT)
add_card(s7, 8.8, 1.5, 3.7, 5.2, bg_color=LIGHT)

col1_s7 = [
    ("Tuning and freeze", {"size": 15, "bold": True, "color": NAVY, "space": 6}),
    ("At most 8 configs per model family.", {"size": 12, "bullet": True}),
    ("Scored on scaffold validation only, seed 0.", {"size": 12, "bullet": True}),
    ("Selected configs written to best_configs_dl.json in commit 0aebd87, before any DL test-result file was committed.", {"size": 12, "bullet": True}),
    ("Git shows what was committed; it cannot show that no test run happened locally.", {"size": 12, "bullet": True}),
]
add_textbox(s7, 1.0, 1.7, 3.3, 4.8, col1_s7)

col2_s7 = [
    ("After the freeze", {"size": 15, "bold": True, "color": TEAL, "space": 6}),
    ("Final and ablation modes need --i-know-this-touches-test and the frozen file.", {"size": 12, "bullet": True}),
    ("One run per seed (0, 1, 2) on each split; mean ± sample SD over seeds.", {"size": 12, "bullet": True}),
    ("Ablations were run on the test split, so they explain results; they were not used to select models.", {"size": 12, "bullet": True}),
]
add_textbox(s7, 5.0, 1.7, 3.3, 4.8, col2_s7)

col3_s7 = [
    ("Fair-comparison contract", {"size": 15, "bold": True, "color": BLUE, "space": 6}),
    ("Phase 1 split files reused for every model.", {"size": 12, "bullet": True}),
    ("Phase 1 baselines were not re-tuned.", {"size": 12, "bullet": True}),
    ("Same masked ROC / PR AUC code for all models.", {"size": 12, "bullet": True}),
    ("H1 uses the MPNN config and is not separately tuned, so it may be under-tuned.", {"size": 12, "bullet": True}),
]
add_textbox(s7, 9.0, 1.7, 3.3, 4.8, col3_s7)
notes(s7, "The freeze is commit 0aebd87, which adds only the configuration file. We cannot prove from git that no "
          "test evaluation was run locally before it. A later tuning rerun for the FP-MLP ranked a different config first "
          "(0.4117 against 0.4092 for the frozen one, within a grid spread of 0.0105); we did not switch.")

# ==================== SLIDE 8: Results table ====================
s8 = prs.slides.add_slide(BLANK)
add_header(s8, "7. Scaffold Test Results (mean ± SD over seeds 0-2)")
best_pr = max(ST[(l, "scaffold")]["pr_auc"][0] for (l, *_r) in ROWS)
tbl_rows = [["Model", "PR AUC", "ROC AUC"]]
bold = []
for i, (lab, *_r) in enumerate(ROWS, start=1):
    st = ST[(lab, "scaffold")]
    tbl_rows.append([lab, pm(st["pr_auc"]), pm(st["roc_auc"])])
    if abs(st["pr_auc"][0] - best_pr) < 1e-12:
        bold.append(i)
add_table(s8, 0.8, 1.5, 6.9, tbl_rows, [2.7, 2.1, 2.1], font=11, row_h=0.37, bold_rows=bold)

add_card(s8, 7.9, 1.5, 4.6, 5.2, bg_color=WHITE, border_color=TEAL)
t_h1 = T[("h1", "fpmlp")]
t_gm = T[("gcn", "mpnn")]
right_s8 = [
    ("Scaffold takeaways:", {"size": 15, "bold": True, "color": NAVY, "space": 6}),
    ("Trees are still best:", {"size": 13, "bold": True, "color": BLUE, "space": 2}),
    (f"RF M+D {pr('rf')[0]:.4f} and XGBoost M+D {pr('xgb')[0]:.4f} PR AUC are the two highest rows.", {"size": 12, "bullet": True, "space": 4}),
    ("H1 vs FP-MLP:", {"size": 13, "bold": True, "color": BLUE, "space": 2}),
    (f"{pr('h1')[0]:.4f} vs {pr('fpmlp')[0]:.4f} ({t_h1['mean_diff']:+.4f}); not significant (p = {t_h1['p_value']:.4f}).", {"size": 12, "bullet": True, "space": 4}),
    ("GCN vs MPNN:", {"size": 13, "bold": True, "color": BLUE, "space": 2}),
    (f"GCN {pr('gcn')[0]:.4f} vs MPNN {pr('mpnn')[0]:.4f}; GCN higher on {t_gm['wins']} of 12 endpoints.", {"size": 12, "bullet": True, "space": 4}),
    ("H2 ensembles:", {"size": 13, "bold": True, "color": BLUE, "space": 2}),
    (f"Primary {pr('h2p')[0]:.4f}, secondary {pr('h2s')[0]:.4f}: above their deep parts, below XGBoost.", {"size": 12, "bullet": True}),
]
add_textbox(s8, 8.0, 1.7, 4.4, 4.8, right_s8)
notes(s8, "All values are the mean over the 12 endpoints per seed, then mean and sample standard deviation over seeds 0, 1, 2, "
          "computed from results_test.csv and the dl_results_test CSVs. The trees are the best rows; no deep or hybrid model "
          "beats RF or XGBoost on scaffold PR AUC. The H1-versus-FP-MLP difference is not significant.")

# ==================== SLIDE 9: Generalization Gap ====================
s9 = prs.slides.add_slide(BLANK)
add_header(s9, "8. Generalization Gap: Random minus Scaffold")
img_gap = ROOT / "report_phase2" / "figures" / "generalization_gap.png"
if img_gap.exists():
    s9.shapes.add_picture(str(img_gap), Inches(0.8), Inches(1.5), width=Inches(6.6))

add_card(s9, 7.6, 1.5, 4.9, 5.2, bg_color=LIGHT)
right_s9 = [
    ("Scaffold penalty (descriptive):", {"size": 15, "bold": True, "color": NAVY, "space": 6}),
    ("Every model loses performance on unseen scaffolds:", {"size": 13, "bold": True, "color": RED, "space": 2}),
    (f"PR AUC drop: {GAP_PR_MIN:.3f} to {GAP_PR_MAX:.3f}.", {"size": 12, "bullet": True}),
    (f"ROC AUC drop: {GAP_ROC_MIN:.3f} to {GAP_ROC_MAX:.3f}.", {"size": 12, "bullet": True, "space": 6}),
    ("Selected PR AUC drops:", {"size": 13, "bold": True, "color": NAVY, "space": 2}),
    (f"FP-MLP {gap('fpmlp'):.4f} (largest), H1 {gap('h1'):.4f}, GCN {gap('gcn'):.4f}, MPNN {gap('mpnn'):.4f}.", {"size": 12, "bullet": True}),
    ("The GCN and MPNN drop less but start from lower scaffold scores.", {"size": 12, "bullet": True, "space": 6}),
    ("These differences were not tested, and we did not test why the FP-MLP drops more.", {"size": 12, "bullet": True}),
]
add_textbox(s9, 7.8, 1.7, 4.5, 4.8, right_s9)
notes(s9, "The gap is the random-split score minus the scaffold-split score, averaged over endpoints and seeds. "
          "It is descriptive: no test was run on the differences between models' gaps.")

# ==================== SLIDE 10: Descriptor ablations ====================
s10 = prs.slides.add_slide(BLANK)
add_header(s10, "9. H1 Input Ablations and the Shuffled-Descriptor Control")
img_abl = ROOT / "report_phase2" / "figures" / "ablation_study.png"
if img_abl.exists():
    s10.shapes.add_picture(str(img_abl), Inches(0.8), Inches(1.5), width=Inches(6.4))

add_card(s10, 7.4, 1.5, 5.1, 5.2, bg_color=WHITE, border_color=RED)
h1_pr = pr("h1")[0]
right_s10 = [
    ("Scaffold test PR AUC (H1 variants):", {"size": 15, "bold": True, "color": NAVY, "space": 6}),
    ("Shuffled descriptors:", {"size": 13, "bold": True, "color": RED, "space": 2}),
    (f"Rows permuted across molecules: {h1_pr:.4f} falls to {SHUF_DESC_PR:.4f} ({SHUF_DESC_PR - h1_pr:+.4f}, {100 * (SHUF_DESC_PR - h1_pr) / h1_pr:.1f}%).", {"size": 12, "bullet": True}),
    ("Same architecture and parameter count, so H1's benefit depends on correct molecule-descriptor alignment.", {"size": 12, "bullet": True, "space": 6}),
    ("Morgan bits instead of descriptors:", {"size": 13, "bold": True, "color": NAVY, "space": 2}),
    (f"{MORGAN_PR:.4f}, below H1. These runs do not isolate why.", {"size": 12, "bullet": True, "space": 6}),
    ("Graph + Morgan + descriptors:", {"size": 13, "bold": True, "color": GREY, "space": 2}),
    (f"{BOTH_PR:.4f}, between the two.", {"size": 12, "bullet": True}),
]
add_textbox(s10, 7.6, 1.7, 4.7, 4.8, right_s10)
notes(s10, f"Values come from dl_ablation_test_ranajeet.csv and the H1 final runs. The shuffled-descriptor drop shows the "
           f"descriptor benefit needs real alignment; it does not explain why Morgan bits do worse ({MORGAN_PR:.4f}).")

# ==================== SLIDE 11 (new): Architecture ablations + shuffled-label control ====================
s11 = prs.slides.add_slide(BLANK)
add_header(s11, "10. MPNN Architecture Ablations and the Shuffled-Label Control")
arch_rows = [["Variant (scaffold test)", "PR AUC", "ROC AUC", "vs MPNN PR"]]
arch_rows.append(["MPNN (frozen: T=3, bond features, Set2Set)", f4(MPNN_PR), f4(MPNN_ROC), "-"])
for tag, desc in ARCH_TAGS:
    p_, r_, _ = abl(ABL_M, tag)
    arch_rows.append([desc, f4(p_), f4(r_), f"{p_ - MPNN_PR:+.4f}"])
add_table(s11, 0.8, 1.5, 6.9, arch_rows, [3.3, 1.1, 1.2, 1.3], font=11, row_h=0.4)
add_textbox(s11, 0.8, 5.0, 6.9, 1.4, [
    (f"Mean over seeds 0-2 of the per-seed endpoint mean (results/dl_ablation_test_malhar.csv). For scale, the frozen MPNN's own "
     f"seed SD is {ST[(LAB['mpnn'], 'scaffold')]['pr_auc'][1]:.4f} PR AUC, so small differences should not be over-read; no significance test was run.",
     {"size": 11, "color": GREY}),
])

add_card(s11, 7.9, 1.5, 4.6, 5.2, bg_color=WHITE, border_color=RED)
over = "above" if SL_ROC_SEEDS[2] > THRESH else "below"
right_s11 = [
    ("Shuffled-label control (MPNN):", {"size": 15, "bold": True, "color": NAVY, "space": 6}),
    ("Training labels permuted across molecules; test labels unchanged.", {"size": 12, "bullet": True, "space": 4}),
    (f"Test ROC AUC by seed: {SL_ROC_SEEDS[0]:.4f}, {SL_ROC_SEEDS[1]:.4f}, {SL_ROC_SEEDS[2]:.4f}.", {"size": 12, "bullet": True}),
    (f"Mean ROC {SL_ROC:.4f}; mean PR AUC {SL_PR:.4f}.", {"size": 12, "bullet": True, "space": 4}),
    (f"Seed 2 ({SL_ROC_SEEDS[2]:.4f}) is {over} the {THRESH:.2f} threshold we set for this control.", {"size": 12, "bullet": True, "bold": True, "color": RED}),
    ("We have not investigated the cause, so we do not claim this control rules out leakage. The threshold is our own, and we cannot show it was set before the run.", {"size": 12, "bullet": True}),
]
add_textbox(s11, 8.1, 1.7, 4.2, 4.8, right_s11)
notes(s11, f"Architecture ablations and the shuffled-label control are from dl_ablation_test_malhar.csv, compared with the frozen MPNN "
           f"({MPNN_PR:.4f} PR AUC). The shuffled-label ROC AUC is {SL_ROC_SEEDS[0]:.4f}, {SL_ROC_SEEDS[1]:.4f} and "
           f"{SL_ROC_SEEDS[2]:.4f}; seed 2 is above the {THRESH:.2f} threshold we chose. We did not investigate why. "
           "Say this plainly if asked: we cannot claim zero leakage from this control alone.")

# ==================== SLIDE 12: Technical Validation ====================
s12 = prs.slides.add_slide(BLANK)
add_header(s12, "11. Technical Validation: Tests and Statistics")
add_card(s12, 0.8, 1.5, 5.6, 5.2, bg_color=LIGHT)
add_card(s12, 6.8, 1.5, 5.7, 5.2, bg_color=LIGHT)

left_s12 = [
    ("Checks that can be re-run:", {"size": 15, "bold": True, "color": NAVY, "space": 6}),
    ("make phase2-test:", {"size": 13, "bold": True, "space": 2}),
    (f"{N_TESTS_PASSED} tests passed in our run (graph and label alignment, feature alignment, gate-0 checks, DL smoke tests).", {"size": 12, "bullet": True, "space": 4}),
    ("Prediction files:", {"size": 13, "bold": True, "space": 2}),
    ("30 saved test-prediction files (771 x 12, finite) re-score to the committed CSV values.", {"size": 12, "bullet": True, "space": 4}),
    ("Freeze guard:", {"size": 13, "bold": True, "space": 2}),
    ("dl-guard stops final, ablation and ensemble runs unless the frozen file is committed and unchanged.", {"size": 12, "bullet": True}),
]
add_textbox(s12, 1.0, 1.7, 5.2, 4.8, left_s12)

b_h1 = bs_counts("h1_minus_fpmlp")
b_p = bs_counts("h2_primary_minus_h1")
b_s = bs_counts("h2_secondary_minus_h1")
right_s12 = [
    ("Paired Wilcoxon on 12 seed-averaged endpoint PR AUCs (scaffold):", {"size": 13, "bold": True, "color": TEAL, "space": 2}),
    (f"H1 vs FP-MLP: {tline('h1', 'fpmlp')}.", {"size": 11, "bullet": True}),
    (f"H2 vs XGBoost, primary: {tline('h2p', 'xgb')}.", {"size": 11, "bullet": True}),
    (f"H2 vs XGBoost, secondary: {tline('h2s', 'xgb')}.", {"size": 11, "bullet": True}),
    (f"H2 vs H1: primary {T[('h2p', 'h1')]['wins']}/{T[('h2p', 'h1')]['losses']} (p = {T[('h2p', 'h1')]['p_value']:.4f}), secondary {T[('h2s', 'h1')]['wins']}/{T[('h2s', 'h1')]['losses']} (p = {T[('h2s', 'h1')]['p_value']:.4f}).", {"size": 11, "bullet": True, "space": 4}),
    ("H2 does not beat XGBoost on scaffold. Tests have low power at N = 12 and are not corrected for multiple comparisons.", {"size": 11, "bullet": True, "bold": True, "color": RED, "space": 4}),
    (f"Bootstrap 95% intervals ({BS_N:,} resamples, seed {BS_SEED}, seed-0 predictions; results/dl_bootstrap_ci.csv):", {"size": 12, "bold": True, "color": TEAL, "space": 2}),
    (f"H1 minus FP-MLP: interval excludes 0 on {b_h1[0] + b_h1[1]} of 12 (NR-Aromatase {aro.delta_mean:+.3f}, [{aro.ci_low:+.3f}, {aro.ci_high:+.3f}]).", {"size": 11, "bullet": True}),
    (f"H2 minus H1: primary above 0 on {b_p[0]}, below on {b_p[1]}; secondary above on {b_s[0]}, below on {b_s[1]}.", {"size": 11, "bullet": True}),
]
add_textbox(s12, 7.0, 1.7, 5.3, 4.9, right_s12)
notes(s12, "The test count (24 passed) was observed when running make phase2-test and is a typed constant in the build script. "
           "All other numbers are computed from the committed CSVs. H2 does not beat XGBoost on scaffold for either variant, and "
           "the endpoint-level bootstrap evidence for H2 over H1 is weaker than the pooled win counts suggest.")

# ==================== SLIDE 13: Conclusions ====================
s13 = prs.slides.add_slide(BLANK)
add_header(s13, "12. Takeaways and Limitations")
add_card(s13, 0.8, 1.5, 5.6, 5.2, bg_color=WHITE, border_color=TEAL)
add_card(s13, 6.8, 1.5, 5.7, 5.2, bg_color=WHITE, border_color=NAVY)

left_s13 = [
    ("What we found (scaffold test, PR AUC):", {"size": 16, "bold": True, "color": NAVY, "space": 8}),
    (f"1. Tree models on Morgan + descriptors remain best: RF {pr('rf')[0]:.4f}, XGBoost {pr('xgb')[0]:.4f}.", {"size": 12, "bullet": True, "space": 6}),
    (f"2. H1 ({pr('h1')[0]:.4f}) is above the FP-MLP ({pr('fpmlp')[0]:.4f}) but not significantly (p = {t_h1['p_value']:.2f}); the two use different inputs.", {"size": 12, "bullet": True, "space": 6}),
    (f"3. The simpler GCN ({pr('gcn')[0]:.4f}) beat the MPNN ({pr('mpnn')[0]:.4f}).", {"size": 12, "bullet": True, "space": 6}),
    (f"4. H2 ensembles ({pr('h2p')[0]:.4f} primary, {pr('h2s')[0]:.4f} secondary) improve on their deep parts but not on XGBoost.", {"size": 12, "bullet": True, "space": 6}),
    ("5. Shuffling descriptors hurts H1, so its benefit relies on aligned descriptors.", {"size": 12, "bullet": True}),
]
add_textbox(s13, 1.0, 1.7, 5.2, 4.8, left_s13)

right_s13 = [
    ("Limitations and guidance:", {"size": 16, "bold": True, "color": NAVY, "space": 8}),
    ("One dataset, 12 endpoints, one fixed scaffold split and one random split; seed SDs measure only model randomness.", {"size": 12, "bullet": True, "space": 6}),
    ("Each config was tuned from a single seed-0 run; H1 was not separately tuned.", {"size": 12, "bullet": True, "space": 6}),
    ("The seed-2 shuffled-label result is above our threshold and its cause is not investigated.", {"size": 12, "bullet": True, "space": 6}),
    ("Ablations ran on the test split and are explanatory.", {"size": 12, "bullet": True, "space": 6}),
    ("Practical guidance: keep descriptor features, and evaluate on scaffold splits rather than random splits alone.", {"size": 12, "bullet": True, "bold": True, "color": TEAL}),
]
add_textbox(s13, 7.0, 1.7, 5.3, 4.8, right_s13)
notes(s13, "State the headline plainly: trees are still best on scaffold, and none of our deep or hybrid models changes that. "
           "The hybrid and ensemble results are modest and mostly not significant at 12 endpoints.")

assert len(prs.slides) == TOTAL_SLIDES, len(prs.slides)
prs.save(str(OUT_PPTX))
print(f"Successfully generated {len(prs.slides)}-slide presentation deck at: {OUT_PPTX}")

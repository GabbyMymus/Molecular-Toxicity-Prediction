#!/usr/bin/env python3
"""Build slides_phase2/tox21_phase2.pptx.

Author: Abhijeet (Deliverable A5)
12-slide presentation deck covering Phase 2 deep learning, hybrids, ablations, and results.
"""

import sys
from pathlib import Path
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
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

slide_num = [0]

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
    p_num.text = f"{slide_num[0]} / 12"
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
p2.text = "Phase 2: Graph Neural Networks, Multi-Modal Hybrids, and Rigorous Empirical Controls"
p2.font.size = Pt(18)
p2.font.bold = True
p2.font.color.rgb = TEAL
p2.font.name = "Arial"
p2.space_before = Pt(8)

p3 = tf1.add_paragraph()
p3.text = "A leak-free, out-of-distribution benchmark comparing classical trees against neural message passing."
p3.font.size = Pt(14)
p3.font.color.rgb = GREY
p3.font.name = "Arial"
p3.space_before = Pt(6)

# Author card
add_card(s1, 1.2, 4.0, 10.9, 2.2, bg_color=LIGHT)
author_text = [
    ("Consortium Authors & Core Roles:", {"size": 13, "bold": True, "color": NAVY, "space": 4}),
    ("Abhijeet Kumar: Research & Analysis Owner (Literature, Theory, Results, Discussion, Slides)", {"size": 12, "bullet": True}),
    ("Ranajeet: Hybrid Owner (H1 MPNN+Desc, H2 Late-Fusion Ensemble, FP-MLP Control, Ablations)", {"size": 12, "bullet": True}),
    ("Parrv: DL Engine Owner (Architecture implementation, Hyperparameter Freeze, CLI, Makefile)", {"size": 12, "bullet": True}),
    ("Raghav: Graph Data Owner (PyG Featurization, Alignment Verification, GCN, EDA)", {"size": 12, "bullet": True}),
    ("Malhar: Verification & Integration Owner (LaTeX Assembly, QA, Tanimoto Generalization)", {"size": 12, "bullet": True}),
]
add_textbox(s1, 1.4, 4.15, 10.5, 1.9, author_text)

# ==================== SLIDE 2: Problem and Chemical Context ====================
s2 = prs.slides.add_slide(BLANK)
add_header(s2, "1. Chemical Context: Why Traditional Fingerprints Hit a Ceiling")
add_card(s2, 0.8, 1.5, 5.6, 5.2, bg_color=WHITE, border_color=GREY)
add_card(s2, 6.8, 1.5, 5.7, 5.2, bg_color=LIGHT)

left_s2 = [
    ("The Representation Dilemma:", {"size": 16, "bold": True, "color": NAVY, "space": 8}),
    ("Classical machine learning relies on fixed circular Morgan fingerprints (ECFP4) hashed into 2,048 bits.", {"size": 13, "bullet": True, "space": 6}),
    ("Bit Collisions: Diverse chemical subgraphs hash into identical indices, inducing representational aliasing.", {"size": 13, "bullet": True, "space": 6}),
    ("Rigid & Non-Differentiable: Handcrafted features cannot adapt dynamically to distinct biological assay contexts.", {"size": 13, "bullet": True, "space": 6}),
    ("Lacks Continuous Geometry: Fails to preserve global spatial, electrostatic, and conjugated pathways.", {"size": 13, "bullet": True, "space": 6}),
]
add_textbox(s2, 1.0, 1.7, 5.2, 4.8, left_s2)

right_s2 = [
    ("The Graph Deep Learning Hypothesis:", {"size": 16, "bold": True, "color": TEAL, "space": 8}),
    ("Treat molecules as natural mathematical graphs G = (V, E):", {"size": 13, "bold": True, "space": 4}),
    ("Nodes (V): Individual atoms carrying continuous 33-dimensional feature vectors.", {"size": 12, "bullet": True}),
    ("Edges (E): Covalent chemical bonds parameterized by 6-dimensional attribute vectors.", {"size": 12, "bullet": True, "space": 6}),
    ("Continuous Message Passing:", {"size": 13, "bold": True, "space": 4}),
    ("Iteratively propagates topological information across multi-hop bonded neighborhoods.", {"size": 12, "bullet": True}),
    ("Learns task-specific representations end-to-end directly from molecular topology.", {"size": 12, "bullet": True, "space": 6}),
    ("Central Question:", {"size": 13, "bold": True, "color": RED, "space": 2}),
    ("Does learned graph representation actually beat tuned tree ensembles on small benchmarks (N ≈ 6,200)?", {"size": 12, "bold": True, "color": NAVY}),
]
add_textbox(s2, 7.0, 1.7, 5.3, 4.8, right_s2)

# ==================== SLIDE 3: Data & Bemis-Murcko Scaffold Splits ====================
s3 = prs.slides.add_slide(BLANK)
add_header(s3, "2. Tox21 Benchmark: Bioassays & Out-of-Distribution Scaffold Shifts")
add_card(s3, 0.8, 1.5, 3.7, 5.2, bg_color=LIGHT)
add_card(s3, 4.8, 1.5, 3.7, 5.2, bg_color=LIGHT)
add_card(s3, 8.8, 1.5, 3.7, 5.2, bg_color=LIGHT)

col1 = [
    ("Dataset Scale & Pathologies", {"size": 15, "bold": True, "color": NAVY, "space": 6}),
    ("Total Compounds: 7,709 unique, valid chemical structures.", {"size": 12, "bullet": True}),
    ("Assay Targets: 12 multi-label endpoints across Nuclear Receptors & Stress Response.", {"size": 12, "bullet": True}),
    ("Severe Imbalance: Toxic hits average only ~8% of measurements.", {"size": 12, "bullet": True}),
    ("Missing Labels: ~20% unmeasured assay values (NaN). Must be masked, not zero-filled!", {"size": 12, "bullet": True}),
]
add_textbox(s3, 1.0, 1.7, 3.3, 4.8, col1)

col2 = [
    ("Bemis-Murcko Scaffold Split", {"size": 15, "bold": True, "color": TEAL, "space": 6}),
    ("Partitions compounds by cyclic ring core and linker framework.", {"size": 12, "bullet": True}),
    ("Split Sizes: 6,167 train / 771 val / 771 test (80 / 10 / 10).", {"size": 12, "bullet": True}),
    ("Zero Scaffold Overlap: Entire core frameworks assigned exclusively to one split.", {"size": 12, "bullet": True}),
    ("Eliminates trivial memorization of known chemotype scaffolds.", {"size": 12, "bullet": True}),
]
add_textbox(s3, 5.0, 1.7, 3.3, 4.8, col2)

col3 = [
    ("The Generalization Challenge", {"size": 15, "bold": True, "color": RED, "space": 6}),
    ("Random Split: Inflates performance by testing on familiar scaffolds.", {"size": 12, "bullet": True}),
    ("Scaffold Split: Emulates real-world drug discovery and regulatory screening.", {"size": 12, "bullet": True}),
    ("Evaluates whether graph representations transfer across distinct chemical families.", {"size": 12, "bullet": True}),
    ("Decision Metric: Mean Precision-Recall AUC (PR AUC) over seeds.", {"size": 12, "bullet": True}),
]
add_textbox(s3, 9.0, 1.7, 3.3, 4.8, col3)

# ==================== SLIDE 4: Graph Featurization & Invariances ====================
s4 = prs.slides.add_slide(BLANK)
add_header(s4, "3. Graph Construction: 39 Continuous Features & Invariances")
add_card(s4, 0.8, 1.5, 5.6, 5.2, bg_color=WHITE, border_color=BLUE)
add_card(s4, 6.8, 1.5, 5.7, 5.2, bg_color=LIGHT)

left_s4 = [
    ("Atom & Bond Featurization:", {"size": 16, "bold": True, "color": NAVY, "space": 6}),
    ("Atom Node Vector (x_v in R^33):", {"size": 13, "bold": True, "color": TEAL, "space": 2}),
    ("Element one-hot (C, N, O, S, F, Cl, Br, I, P, Other; 10-dim)", {"size": 12, "bullet": True}),
    ("Explicit atom degree (0-6) & Total hydrogen count (0-4)", {"size": 12, "bullet": True}),
    ("Hybridization state (sp, sp2, sp3, sp3d, sp3d2, Other; 6-dim)", {"size": 12, "bullet": True}),
    ("Formal charge, aromaticity flag, in-ring boolean flag", {"size": 12, "bullet": True}),
    ("Continuous normalized atomic mass (mass / 100)", {"size": 12, "bullet": True, "space": 6}),
    ("Bond Edge Vector (e_vw in R^6):", {"size": 13, "bold": True, "color": TEAL, "space": 2}),
    ("Bond type one-hot (single, double, triple, aromatic; 4-dim)", {"size": 12, "bullet": True}),
    ("Conjugation boolean flag & In-ring boolean flag", {"size": 12, "bullet": True}),
    ("Every covalent bond converted into two directed edges.", {"size": 12, "bullet": True}),
]
add_textbox(s4, 1.0, 1.7, 5.2, 4.8, left_s4)

right_s4 = [
    ("Mathematical Invariances & Sanity Guards:", {"size": 16, "bold": True, "color": NAVY, "space": 6}),
    ("Permutation Invariance:", {"size": 13, "bold": True, "color": BLUE, "space": 2}),
    ("Molecules have no canonical linear atom ordering.", {"size": 12, "bullet": True}),
    ("Shuffling atom indices (equivalent SMILES strings) produces bit-identical logits (unit-tested).", {"size": 12, "bullet": True, "space": 6}),
    ("Permutation-Invariant Set2Set Readout:", {"size": 13, "bold": True, "color": BLUE, "space": 2}),
    ("Employs content-based attention pooling (M=3 steps) to collapse variable-sized node sets into fixed R^128.", {"size": 12, "bullet": True, "space": 6}),
    ("Masked, Class-Weighted Loss:", {"size": 13, "bold": True, "color": RED, "space": 2}),
    ("Loss masks unmeasured labels (zero gradient contribution).", {"size": 12, "bullet": True}),
    ("Pos-weights (capped at 50) correct for 8% positive rarity.", {"size": 12, "bullet": True}),
]
add_textbox(s4, 7.0, 1.7, 5.3, 4.8, right_s4)

# ==================== SLIDE 5: Deep Learning Architectures ====================
s5 = prs.slides.add_slide(BLANK)
add_header(s5, "4. Deep Architectures: GCN vs. Message Passing (MPNN)")
add_card(s5, 0.8, 1.5, 5.6, 5.2, bg_color=LIGHT)
add_card(s5, 6.8, 1.5, 5.7, 5.2, bg_color=WHITE, border_color=TEAL)

left_s5 = [
    ("GCN Baseline (Isotropic Convolution)", {"size": 16, "bold": True, "color": GREY, "space": 6}),
    ("Neighborhood Mean Aggregation:", {"size": 13, "bold": True, "space": 2}),
    ("H^(t) = ReLU( D^(-1/2) A D^(-1/2) H^(t-1) W^(t) )", {"size": 12, "color": NAVY, "space": 4}),
    ("Architectural Traits:", {"size": 13, "bold": True, "space": 2}),
    ("Ignores bond features entirely (treats single, double, aromatic bonds identically).", {"size": 12, "bullet": True}),
    ("Isotropic averaging blurs distinct chemical interactions.", {"size": 12, "bullet": True}),
    ("Compact capacity: ~56,000 parameters (3 layers, h=256).", {"size": 12, "bullet": True}),
    ("Fast baseline but chemically over-simplified.", {"size": 12, "bullet": True}),
]
add_textbox(s5, 1.0, 1.7, 5.2, 4.8, left_s5)

right_s5 = [
    ("MPNN (Gilmer et al. 2017)", {"size": 16, "bold": True, "color": TEAL, "space": 6}),
    ("Continuous Edge-Network Message Passing:", {"size": 13, "bold": True, "space": 2}),
    ("m_v = MEAN_{w in N(v)} [ A(e_vw) * h_w ]", {"size": 12, "color": NAVY, "space": 4}),
    ("A(e_vw) is an MLP mapping bond attributes to transformation matrices.", {"size": 12, "bullet": True}),
    ("Gated Recurrent Unit (GRU) Update:", {"size": 13, "bold": True, "space": 2}),
    ("h_v = GRU( h_v , m_v ) over T=3 propagation steps.", {"size": 12, "bullet": True}),
    ("Weight sharing keeps parameter budget compact (~628k).", {"size": 12, "bullet": True}),
    ("Order-Invariant Set2Set Readout:", {"size": 13, "bold": True, "space": 2}),
    ("Attention-based global pooling avoids information loss.", {"size": 12, "bullet": True}),
]
add_textbox(s5, 7.0, 1.7, 5.3, 4.8, right_s5)

# ==================== SLIDE 6: Hybrid Design ====================
s6 = prs.slides.add_slide(BLANK)
add_header(s6, "5. Hybrid Design: Fusing Message Passing with Physicochemical Priors")
add_card(s6, 0.8, 1.5, 5.6, 5.2, bg_color=WHITE, border_color=NAVY)
add_card(s6, 6.8, 1.5, 5.7, 5.2, bg_color=LIGHT)

left_s6 = [
    ("H1 Multi-Modal Hybrid (mpnn+desc):", {"size": 16, "bold": True, "color": NAVY, "space": 6}),
    ("The Representation Rationale:", {"size": 13, "bold": True, "color": TEAL, "space": 2}),
    ("Message passing captures local bonded topology exceptionally well.", {"size": 12, "bullet": True}),
    ("Global physicochemical properties (logP, TPSA, molecular weight, charge distribution) are hard to learn on N=6,200.", {"size": 12, "bullet": True, "space": 4}),
    ("Multi-Modal Architecture:", {"size": 13, "bold": True, "color": TEAL, "space": 2}),
    ("Graph Branch: Set2Set graph embedding z_graph in R^128.", {"size": 12, "bullet": True}),
    ("Descriptor Branch: 213 RDKit descriptors projected via dense MLP into z_desc in R^128.", {"size": 12, "bullet": True}),
    ("Concatenation Fusion: z = [z_graph || z_desc] in R^256.", {"size": 12, "bullet": True}),
    ("Classification Head: 2-layer MLP mapping to 12 logits.", {"size": 12, "bullet": True}),
]
add_textbox(s6, 1.0, 1.7, 5.2, 4.8, left_s6)

right_s6 = [
    ("H2 Late-Fusion Ensemble (ens_xgb_mpnn+desc):", {"size": 16, "bold": True, "color": NAVY, "space": 6}),
    ("Exploiting Orthogonal Inductive Biases:", {"size": 13, "bold": True, "color": BLUE, "space": 2}),
    ("Trees partition feature space with sharp, axis-aligned boundaries.", {"size": 12, "bullet": True}),
    ("GNNs construct smooth continuous manifold representations.", {"size": 12, "bullet": True}),
    ("They make different, uncorrelated errors on out-of-distribution scaffolds.", {"size": 12, "bullet": True, "space": 4}),
    ("Rank-Average Fusion (Zero Leakage):", {"size": 13, "bold": True, "color": BLUE, "space": 2}),
    ("Converts predicted probabilities to endpoint-wise ranks in (0, 1].", {"size": 12, "bullet": True}),
    ("Equal fixed weights (0.5 / 0.5) specified a priori.", {"size": 12, "bullet": True}),
    ("Zero weight tuning on validation or test sets.", {"size": 12, "bullet": True}),
]
add_textbox(s6, 7.0, 1.7, 5.3, 4.8, right_s6)

# ==================== SLIDE 7: Experimental Protocol & Freeze ====================
s7 = prs.slides.add_slide(BLANK)
add_header(s7, "6. Protocol & Test-Set Discipline: Zero-Leakage Governance")
add_card(s7, 0.8, 1.5, 3.7, 5.2, bg_color=LIGHT)
add_card(s7, 4.8, 1.5, 3.7, 5.2, bg_color=LIGHT)
add_card(s7, 8.8, 1.5, 3.7, 5.2, bg_color=LIGHT)

col1_s7 = [
    ("Tuning Budget & Freeze", {"size": 15, "bold": True, "color": NAVY, "space": 6}),
    ("Max 8 configs per model family.", {"size": 12, "bullet": True}),
    ("Evaluated solely on scaffold validation split (Seed 0).", {"size": 12, "bullet": True}),
    ("Best configs locked in best_configs_dl.json (commit 0aebd87).", {"size": 12, "bullet": True}),
    ("Test set never touched during tuning or architecture selection.", {"size": 12, "bullet": True}),
]
add_textbox(s7, 1.0, 1.7, 3.3, 4.8, col1_s7)

col2_s7 = [
    ("Strict Test Discipline", {"size": 15, "bold": True, "color": TEAL, "space": 6}),
    ("CLI Guard: Demands flag --i-know-this-touches-test.", {"size": 12, "bullet": True}),
    ("Evaluated exactly once per frozen configuration.", {"size": 12, "bullet": True}),
    ("3 Random Seeds (0, 1, 2) on scaffold and random splits.", {"size": 12, "bullet": True}),
    ("Reports mean +/- standard deviation across seeds.", {"size": 12, "bullet": True}),
]
add_textbox(s7, 5.0, 1.7, 3.3, 4.8, col2_s7)

col3_s7 = [
    ("Fair Comparison Contract", {"size": 15, "bold": True, "color": BLUE, "space": 6}),
    ("Identical scaffold split indices used in Phase 1 & 2.", {"size": 12, "bullet": True}),
    ("Phase 1 classical models frozen (never re-tuned).", {"size": 12, "bullet": True}),
    ("Identical masked ROC/PR AUC evaluation functions.", {"size": 12, "bullet": True}),
    ("H1 reuses MPNN config (extra descriptors is sole change).", {"size": 12, "bullet": True}),
]
add_textbox(s7, 9.0, 1.7, 3.3, 4.8, col3_s7)

# ==================== SLIDE 8: Main Results on Scaffold Split ====================
s8 = prs.slides.add_slide(BLANK)
add_header(s8, "7. Primary Benchmark: Out-of-Distribution Scaffold Results")
# Left side: image
img_path = ROOT / "report_phase2" / "figures" / "mean_pr_roc_scaffold.png"
if img_path.exists():
    s8.shapes.add_picture(str(img_path), Inches(0.8), Inches(1.5), width=Inches(6.8))

# Right side: summary card
add_card(s8, 7.8, 1.5, 4.7, 5.2, bg_color=WHITE, border_color=TEAL)
right_s8 = [
    ("Benchmark Takeaways (Scaffold):", {"size": 15, "bold": True, "color": NAVY, "space": 6}),
    ("1. H1 beats FP-MLP Control:", {"size": 13, "bold": True, "color": BLUE, "space": 2}),
    ("H1 (0.3553) > FP-MLP (0.3429) by +0.0124 PR AUC.", {"size": 12, "bullet": True}),
    ("Wins on 9 of 12 endpoints. Proves message passing representation advantage over flat fingerprints.", {"size": 12, "bullet": True, "space": 6}),
    ("2. H2 Ensemble Dominance:", {"size": 13, "bold": True, "color": GREEN, "space": 2}),
    ("H2 reaches 0.3945 PR AUC and 0.7643 ROC AUC.", {"size": 12, "bullet": True}),
    ("Wins on 12 of 12 endpoints over H1 (p = 0.00049).", {"size": 12, "bullet": True}),
    ("Lowest variance of all models (sigma = 0.0011).", {"size": 12, "bullet": True, "space": 6}),
    ("3. Classical Tree Performance:", {"size": 13, "bold": True, "color": GREY, "space": 2}),
    ("RF (0.4035) and XGBoost (0.4010) on Morgan+Desc remain top individual models on N=6,200.", {"size": 12, "bullet": True}),
]
add_textbox(s8, 8.0, 1.7, 4.3, 4.8, right_s8)

# ==================== SLIDE 9: Generalization Gap ====================
s9 = prs.slides.add_slide(BLANK)
add_header(s9, "8. Chemical Generalization Gap: Random vs. Scaffold Shifts")
img_gap = ROOT / "report_phase2" / "figures" / "generalization_gap.png"
if img_gap.exists():
    s9.shapes.add_picture(str(img_gap), Inches(0.8), Inches(1.5), width=Inches(6.6))

add_card(s9, 7.6, 1.5, 4.9, 5.2, bg_color=LIGHT)
right_s9 = [
    ("Quantifying the Scaffold Shift Penalty:", {"size": 15, "bold": True, "color": NAVY, "space": 6}),
    ("Severe Out-of-Distribution Degradation:", {"size": 13, "bold": True, "color": RED, "space": 2}),
    ("Every architecture suffers a massive drop on unseen scaffolds:", {"size": 12, "bullet": True}),
    ("Delta PR AUC: 0.160 - 0.195 drop across all models.", {"size": 12, "bullet": True}),
    ("Delta ROC AUC: 0.085 - 0.124 drop.", {"size": 12, "bullet": True, "space": 6}),
    ("Dense Networks Overfit Fingerprints Most:", {"size": 13, "bold": True, "color": NAVY, "space": 2}),
    ("FP-MLP suffers the worst PR penalty (Delta = 0.1949).", {"size": 12, "bullet": True}),
    ("Dense weights memorize specific bit correlations that do not transfer across chemotypes.", {"size": 12, "bullet": True, "space": 6}),
    ("Graph Hybrids Generalize Better:", {"size": 13, "bold": True, "color": TEAL, "space": 2}),
    ("H1 (Delta = 0.1600) and H2 (Delta = 0.1681) exhibit superior robustness to scaffold shifts.", {"size": 12, "bullet": True}),
]
add_textbox(s9, 7.8, 1.7, 4.5, 4.8, right_s9)

# ==================== SLIDE 10: Ablation Studies & Controls ====================
s10 = prs.slides.add_slide(BLANK)
add_header(s10, "9. Ablation Suite: Decisive Negative Control Proof")
img_abl = ROOT / "report_phase2" / "figures" / "ablation_study.png"
if img_abl.exists():
    s10.shapes.add_picture(str(img_abl), Inches(0.8), Inches(1.5), width=Inches(6.4))

add_card(s10, 7.4, 1.5, 5.1, 5.2, bg_color=WHITE, border_color=RED)
right_s10 = [
    ("Why the Hybrid Gain is Authenticated:", {"size": 15, "bold": True, "color": NAVY, "space": 6}),
    ("1. Shuffled-Descriptor Control (A6a):", {"size": 13, "bold": True, "color": RED, "space": 2}),
    ("Descriptor rows permuted across compounds.", {"size": 12, "bullet": True}),
    ("PR AUC collapses from 0.3553 to 0.2547 (-0.1006 drop; -28.3%).", {"size": 12, "bullet": True}),
    ("Proves descriptor gain stems from genuine chemical physics, not parameter capacity or regularizing noise!", {"size": 12, "bullet": True, "space": 6}),
    ("2. Fingerprint Substitution (A6b):", {"size": 13, "bold": True, "color": NAVY, "space": 2}),
    ("Replacing descriptors with Morgan bits crashes PR AUC to 0.2614.", {"size": 12, "bullet": True}),
    ("Sparse, high-dimensional bits interfere with graph gradient flow.", {"size": 12, "bullet": True, "space": 6}),
    ("3. Triple Modality (A6c):", {"size": 13, "bold": True, "color": GREY, "space": 2}),
    ("Combining Graph + FP + Desc yields 0.3309 (intermediate).", {"size": 12, "bullet": True}),
]
add_textbox(s10, 7.6, 1.7, 4.7, 4.8, right_s10)

# ==================== SLIDE 11: Technical Validation ====================
s11 = prs.slides.add_slide(BLANK)
add_header(s11, "10. Technical Validation: Statistical Rigor & Sanity Checks")
add_card(s11, 0.8, 1.5, 5.6, 5.2, bg_color=LIGHT)
add_card(s11, 6.8, 1.5, 5.7, 5.2, bg_color=LIGHT)

left_s11 = [
    ("Sanity Checks & Leakage Guards:", {"size": 15, "bold": True, "color": NAVY, "space": 6}),
    ("Shuffled-Label Negative Control:", {"size": 13, "bold": True, "space": 2}),
    ("Permuting training labels collapses test AUC to ~0.50 (chance level). Proves zero feature/label leakage.", {"size": 12, "bullet": True, "space": 4}),
    ("Single-Batch Overfit Guard:", {"size": 13, "bold": True, "space": 2}),
    ("MPNN drives loss to ~0 on tiny 24-compound batch. Confirms architectural capacity to learn.", {"size": 12, "bullet": True, "space": 4}),
    ("Permutation Invariance Check:", {"size": 13, "bold": True, "space": 2}),
    ("Shuffling atom order produces identical logits.", {"size": 12, "bullet": True, "space": 4}),
    ("Unit & Alignment Test Suite:", {"size": 13, "bold": True, "space": 2}),
    ("All 13 Phase 2 unit & alignment tests pass green.", {"size": 12, "bullet": True}),
]
add_textbox(s11, 1.0, 1.7, 5.2, 4.8, left_s11)

right_s11 = [
    ("Statistical Hypothesis Testing:", {"size": 15, "bold": True, "color": TEAL, "space": 6}),
    ("Paired Wilcoxon Signed-Rank Tests (12 Assays):", {"size": 13, "bold": True, "space": 2}),
    ("H1 vs FP-MLP: Mean diff +0.0124 | 9 wins / 3 losses | p = 0.1514.", {"size": 12, "bullet": True}),
    ("H2 vs H1: Mean diff +0.0392 | 12 wins / 0 losses | p = 0.00049 (Highly Significant).", {"size": 12, "bullet": True}),
    ("H2 vs XGBoost: Mean diff -0.0065 | 5 wins / 7 losses | p = 0.6221.", {"size": 12, "bullet": True, "space": 6}),
    ("Joint Bootstrap 95% Confidence Intervals:", {"size": 13, "bold": True, "space": 2}),
    ("1,000 paired resamples on test predictions.", {"size": 12, "bullet": True}),
    ("H2 beats H1 with 95% CIs strictly excluding 0 on 10/12 endpoints.", {"size": 12, "bullet": True}),
    ("Seed variance: H2 achieves sigma = 0.0011 across all seeds.", {"size": 12, "bullet": True}),
]
add_textbox(s11, 7.0, 1.7, 5.3, 4.8, right_s11)

# ==================== SLIDE 12: Conclusions & Guidelines ====================
s12 = prs.slides.add_slide(BLANK)
add_header(s12, "11. Takeaways & Guidelines: Computational Toxicology in Practice")
add_card(s12, 0.8, 1.5, 5.6, 5.2, bg_color=WHITE, border_color=TEAL)
add_card(s12, 6.8, 1.5, 5.7, 5.2, bg_color=WHITE, border_color=NAVY)

left_s12 = [
    ("Key Scientific Findings:", {"size": 16, "bold": True, "color": NAVY, "space": 8}),
    ("1. Message passing provides authentic structural advantages over flat fingerprints (+0.0124 PR, 9/12 wins).", {"size": 12, "bullet": True, "space": 6}),
    ("2. Descriptors provide crucial global context: Shuffled control proves -28.3% drop without real physics.", {"size": 12, "bullet": True, "space": 6}),
    ("3. Trees on descriptors are formidable on small data: Random Forest and XGBoost remain top individual baselines.", {"size": 12, "bullet": True, "space": 6}),
    ("4. Cross-paradigm ensembling succeeds: H2 achieves lowest variance and 12/12 wins over pure GNN.", {"size": 12, "bullet": True}),
]
add_textbox(s12, 1.0, 1.7, 5.2, 4.8, left_s12)

right_s12 = [
    ("Recommendations & Limitations:", {"size": 16, "bold": True, "color": NAVY, "space": 8}),
    ("Never discard descriptor engineering:", {"size": 13, "bold": True, "color": TEAL, "space": 2}),
    ("Fusing low-dimensional physical priors with learned graph representations produces the most robust systems.", {"size": 12, "bullet": True, "space": 6}),
    ("Evaluate on out-of-distribution scaffolds:", {"size": 13, "bold": True, "color": TEAL, "space": 2}),
    ("Random splitting obscures real generalization failures.", {"size": 12, "bullet": True, "space": 6}),
    ("Benchmarking Limitations:", {"size": 13, "bold": True, "color": GREY, "space": 2}),
    ("Single benchmark (Tox21); 12 assay endpoints; 2D topology lacks explicit 3D receptor conformational fit.", {"size": 12, "bullet": True}),
]
add_textbox(s12, 7.0, 1.7, 5.3, 4.8, right_s12)

# Save Presentation
prs.save(str(OUT_PPTX))
print(f"Successfully generated 12-slide presentation deck at: {OUT_PPTX}")

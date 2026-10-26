"""Build slides/tox21_phase1.pptx from committed results, tables and figures.

Usage (from tox21-phase1/): python slides/build_deck.py . slides/tox21_phase1.pptx  (needs python-pptx, pandas, pillow).
"""
import json
import sys
from zipfile import ZIP_DEFLATED, ZipFile
from pathlib import Path
from xml.dom import minidom

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(sys.argv[1])  # .../tox21-phase1
OUT = Path(sys.argv[2])

NAVY = RGBColor(0x14, 0x2B, 0x4D)
TEAL = RGBColor(0x1B, 0x7F, 0x8C)
GREY = RGBColor(0x55, 0x5F, 0x6B)
LIGHT = RGBColor(0xEA, 0xF3, 0xF5)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xB2, 0x3A, 0x3A)

W, H = 13.333, 7.5
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]
count = [0]

# ---------- data (all from committed files) ----------
res = pd.read_csv(ROOT / "results/results_test.csv")
cfg = json.load(open(ROOT / "results/best_configs.json"))
summ = pd.read_csv(ROOT / "tables/results_summary_scaffold.csv")
pen = pd.read_csv(ROOT / "tables/scaffold_penalty.csv")
desc = pd.read_csv(ROOT / "tables/descriptor_benefit.csv")
ep = pd.read_csv(ROOT / "tables/endpoint_results_scaffold.csv")
assert len(res) == 432


def text(slide, x, y, w, h, paras, size=18, color=NAVY, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """paras: list of str or (str, dict(size,bold,color,bullet))."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    first = True
    for p in paras:
        t, o = (p, {}) if isinstance(p, str) else p
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        bullet = o.get("bullet", False)
        run = para.add_run()
        run.text = ("•  " + t) if bullet else t
        run.font.size = Pt(o.get("size", size))
        run.font.bold = o.get("bold", bold)
        run.font.color.rgb = o.get("color", color)
        run.font.name = "Arial"
        para.alignment = align
        para.space_after = Pt(o.get("space", 8))
    return tb


def new_slide(title, notes):
    s = prs.slides.add_slide(BLANK)
    count[0] += 1
    text(s, 0.6, 0.3, 12.1, 0.9, [title], size=30, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.65), Inches(1.15), Inches(1.6), Inches(0.06))
    bar.fill.solid(); bar.fill.fore_color.rgb = TEAL; bar.line.fill.background()
    text(s, 0.6, 7.05, 9, 0.3, ["Tox21 Phase 1  |  classical ML baseline"], size=10, color=GREY)
    text(s, 11.9, 7.05, 0.9, 0.3, [str(count[0])], size=10, color=GREY, align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes
    return s


def picture(slide, path, x, y, w, h):
    iw, ih = Image.open(path).size
    r = min(w / iw, h / ih)
    pw, ph = iw * r, ih * r
    slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def table(slide, x, y, w, rows, col_w, size=14, row_h=0.4, bold_cells=(), first_col_left=True):
    nr, nc = len(rows), len(rows[0])
    gs = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    t = gs.table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i in range(nr):
        t.rows[i].height = Inches(row_h)
        for j in range(nc):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            r = p.add_run(); r.text = str(rows[i][j])
            r.font.size = Pt(size); r.font.name = "Arial"
            r.font.bold = (i == 0) or ((i, j) in bold_cells)
            p.alignment = PP_ALIGN.LEFT if (j == 0 and first_col_left) or j < 0 else PP_ALIGN.CENTER
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = NAVY; r.font.color.rgb = WHITE
            elif (i, j) in bold_cells:
                c.fill.fore_color.rgb = RGBColor(0xD8, 0xEF, 0xEA); r.font.color.rgb = NAVY
            else:
                c.fill.fore_color.rgb = LIGHT if i % 2 else WHITE; r.font.color.rgb = NAVY
    return t


def box(slide, x, y, w, h, label, fill=TEAL, size=15):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill; sh.line.fill.background()
    tf = sh.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.08)
    lines = label.split("\n")
    for k, ln in enumerate(lines):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = ln; r.font.name = "Arial"; r.font.color.rgb = WHITE
        r.font.size = Pt(size if k == 0 else size - 3); r.font.bold = (k == 0)
    return sh


def arrow(slide, x, y, w=0.35):
    a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(0.3))
    a.fill.solid(); a.fill.fore_color.rgb = GREY; a.line.fill.background()


def f4(v): return f"{v:.4f}"


def polish_package(path):
    """Apply portable type and project colors to the Office XML package."""
    temp = path.with_name(path.stem + ".polished.pptx")
    colors = {
        "dk1": "142B4D", "lt1": "FFFFFF", "dk2": "142B4D", "lt2": "EAF3F5",
        "accent1": "1B7F8C", "accent2": "B23A3A", "accent3": "555F6B",
        "accent4": "BFE3E8", "accent5": "88B8BD", "accent6": "F2C9C4",
    }
    with ZipFile(path) as source, ZipFile(temp, "w", ZIP_DEFLATED) as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename in {"ppt/theme/theme1.xml", "docProps/core.xml"} or item.filename.startswith(("ppt/slides/slide", "ppt/notesSlides/notesSlide")) and item.filename.endswith(".xml"):
                doc = minidom.parseString(data)
                if item.filename == "ppt/theme/theme1.xml":
                    for node in doc.getElementsByTagName("a:latin"):
                        node.setAttribute("typeface", "Arial")
                    schemes = doc.getElementsByTagName("a:clrScheme")
                    if schemes:
                        for name, value in colors.items():
                            entries = schemes[0].getElementsByTagName(f"a:{name}")
                            if not entries:
                                continue
                            entry = entries[0]
                            for child in list(entry.childNodes):
                                if child.nodeType == child.ELEMENT_NODE:
                                    entry.removeChild(child)
                            color = doc.createElement("a:srgbClr")
                            color.setAttribute("val", value)
                            entry.appendChild(color)
                elif item.filename == "docProps/core.xml":
                    metadata = {
                        "dc:title": "Tox21 Phase 1: Molecular Toxicity Prediction",
                        "dc:subject": "Reproducible classical machine learning benchmark on 12 Tox21 assays",
                        "dc:creator": "Molecular Toxicity Prediction Team",
                        "cp:keywords": "Tox21, toxicity prediction, molecular fingerprints, RDKit, machine learning",
                        "dc:description": "Phase 1 benchmark comparing Logistic Regression, Random Forest and XGBoost with Morgan fingerprints and RDKit descriptors under scaffold and random splits.",
                        "cp:lastModifiedBy": "Molecular Toxicity Prediction Team",
                        "cp:category": "Computational toxicology",
                    }
                    for tag, value in metadata.items():
                        nodes = doc.getElementsByTagName(tag)
                        if nodes:
                            node = nodes[0]
                            while node.firstChild:
                                node.removeChild(node.firstChild)
                            node.appendChild(doc.createTextNode(value))
                else:
                    for node in doc.getElementsByTagName("a:latin"):
                        node.setAttribute("typeface", "Arial")
                    for node in list(doc.getElementsByTagName("a:rPr")) + list(doc.getElementsByTagName("a:endParaRPr")):
                        node.setAttribute("dirty", "0")
                data = doc.toxml(encoding="UTF-8")
            target.writestr(item.filename, data)
    temp.replace(path)


# ---------- 1 title ----------
s = prs.slides.add_slide(BLANK); count[0] += 1
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
accent = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.62), Inches(1.92), Inches(0.08), Inches(3.9))
accent.fill.solid(); accent.fill.fore_color.rgb = TEAL; accent.line.fill.background()
text(s, 0.9, 1.9, 11.5, 1.8, ["Predicting Tox21 toxicity from molecular structure"], size=44, color=WHITE, bold=True)
text(s, 0.9, 3.8, 11.5, 1.2, ["Phase 1: Morgan fingerprints, RDKit descriptors and classical ML\n(Logistic Regression, Random Forest, XGBoost)"], size=22, color=RGBColor(0xBF, 0xE3, 0xE8))
text(s, 0.9, 5.15, 11.5, 0.45, ["7,709 molecules     •     12 assay endpoints     •     3 model families"], size=17, color=WHITE, bold=True)
text(s, 0.9, 5.95, 11.5, 0.5, ["Raghav  |  Ranajeet  |  Parrv  |  Abhijeet  |  Malhar"], size=16, color=RGBColor(0xBF, 0xE3, 0xE8))
s.notes_slide.notes_text_frame.text = (
    "Opening. We predict the outcome of 12 Tox21 toxicity assays from a molecule's structure. Phase 1 is the classical machine learning baseline "
    "and the evaluation machinery; the graph neural network is Phase 2. Team: Raghav (data, splits, Logistic Regression), Ranajeet (features, Random Forest), "
    "Parrv (repository, evaluation harness, XGBoost, tuning and final test), Abhijeet (report and analysis); Malhar is Member 5 (QA and verification); the role is assigned and the QA tasks were completed by Parrv. "
    "Source: README team table.")

# ---------- 2 problem and dataset ----------
s = new_slide("Problem and dataset", (
    "Tox21 gives a SMILES string and 12 binary assay labels per molecule. After cleaning (largest fragment, canonical SMILES, deduplication, conflicting labels set to missing) "
    "there are 7,709 unique molecules. A blank label means the compound was not tested on that assay, not that it is non-toxic, so missing labels stay NaN and every metric uses labeled rows only. "
    "17.05% of all label cells are missing (computed from data/processed/tox21_clean.csv). Labeled molecules per endpoint run from 5,718 to 7,141. "
    "Positive rates run from 2.91% (NR-PPAR-gamma) to 15.94% (SR-ARE), so positives are rare. The figure is figures/eda_labels.png. "
    "Say clearly: we never fill missing labels with 0."))
text(s, 0.6, 1.5, 5.6, 5.4, [
    ("7,709 unique molecules after cleaning", {"bullet": True}),
    ("12 toxicity endpoints (NR-* nuclear receptor, SR-* stress response), one binary classifier per endpoint", {"bullet": True}),
    ("17.05% of label cells are missing: not tested, not non-toxic. Missing labels stay NaN; metrics use labeled rows only", {"bullet": True}),
    ("Labeled molecules per endpoint: 5,718 to 7,141", {"bullet": True}),
    ("Positive rate per endpoint: 2.91% (NR-PPAR-gamma) to 15.94% (SR-ARE)", {"bullet": True}),
], size=19)
picture(s, ROOT / "figures/eda_labels.png", 6.4, 1.9, 6.4, 4.2)

# ---------- 3 pipeline ----------
s = new_slide("Pipeline", (
    "Walk through left to right. Raw Tox21 is cleaned to 7,709 molecules, then split 80/10/10 into 6,167 train, 771 validation and 771 test, with a scaffold split and a random split. "
    "Features are Morgan fingerprints, optionally plus RDKit descriptors. Tuning uses only the validation split. The winning configurations are written to best_configs.json and committed "
    "(commit 041892c) before the final test run. The test split is read only by scripts/04_final_test.py. Each stage has one make target: make data, features, tune, final-test, figures."))
y1, y2 = 1.8, 4.1
bw, bh = 2.55, 1.35
xs = [0.6, 3.75, 6.9, 10.05]
b1 = [("Raw Tox21 CSV", "12 labels, SMILES"), ("Clean + deduplicate", "7,709 molecules"),
      ("Split 80/10/10", "6,167 / 771 / 771\nscaffold + random"), ("Features", "Morgan 2,048 bits\n+ 213 descriptors")]
b2 = [("Tune on validation", "8 configs per model\nper feature set"), ("Freeze configs", "best_configs.json\ncommit 041892c"),
      ("Final test", "seeds 0, 1, 2\n432 classifiers"), ("Tables, figures, report", "from results_test.csv")]
for i, (a, b) in enumerate(b1):
    box(s, xs[i], y1, bw, bh, a + "\n" + b)
    if i < 3: arrow(s, xs[i] + bw + 0.02, y1 + 0.52)
for i, (a, b) in enumerate(b2):
    box(s, xs[i], y2, bw, bh, a + "\n" + b, fill=NAVY if i != 1 else RED)
    if i < 3: arrow(s, xs[i] + bw + 0.02, y2 + 0.52)
text(s, 0.6, 5.9, 12.1, 0.9, [("The test split is read only by scripts/04_final_test.py. Rows are never reordered; missing labels are never filled with 0.", {"size": 17, "color": GREY})])

# ---------- 4 features ----------
s = new_slide("Features: Morgan fingerprints and RDKit descriptors", (
    "Two feature sets. morgan: 2,048-bit Morgan (ECFP4-like, radius 2) fingerprints. morgan+desc: the same plus RDKit 2D descriptors, 2,261 columns in total. "
    "There are 217 raw descriptors; 213 are retained after dropping three that are constant zero across all molecules (SMR_VSA8, SlogP_VSA9, fr_diazo) and fr_azide, which is near-constant in scaffold training. "
    "Descriptor cleaning (infinities, NaNs, scaling) is fit on the training rows only inside load_features(), so no validation or test information leaks. "
    "Numbers on NaN and infinite cells come from the feature notebook documented in the README (reported by the feature owner, Ranajeet)."))
box(s, 0.6, 1.7, 5.8, 1.5, "morgan\n2,048 bits, radius 2", fill=TEAL, size=22)
box(s, 6.9, 1.7, 5.8, 1.5, "morgan+desc\n2,261 columns = 2,048 + 213 descriptors", fill=NAVY, size=22)
text(s, 0.6, 3.6, 12.1, 3.2, [
    ("RDKit 2D descriptors: 217 raw, 213 retained (3 constant-zero and 1 near-constant descriptor dropped)", {"bullet": True}),
    ("Raw descriptors had 1,224 NaN cells and 6 infinite cells across 107 compounds; these are handled before scaling", {"bullet": True}),
    ("Cleaning and scaling are fit on the training rows only, inside load_features(): no leakage into validation or test", {"bullet": True}),
    ("Morgan bits are binary substructure indicators; descriptors add bulk properties such as logP, polar surface area and molecular weight", {"bullet": True}),
], size=19)

# ---------- 5 models and tuning ----------
s = new_slide("Models and tuning protocol", (
    "Three models, one classifier per endpoint, trained on that endpoint's labeled training rows: Logistic Regression, Random Forest and XGBoost. "
    "Class imbalance: LR class_weight balanced, RF balanced_subsample, XGBoost scale_pos_weight = negatives / positives per endpoint. "
    "Tuning: 8 configurations per model and feature set, chosen by mean validation PR AUC across endpoints, seed 0, scaffold split; the chosen configuration is reused unchanged on the random split. "
    "The table shows the chosen configurations from results/best_configs.json. "
    "Freeze proof: best_configs.json was committed in 041892c on 2026-10-25 at 19:41:39 +05:30, and the final test results in 043c2ed at 19:52:52; 041892c is an ancestor of 043c2ed. "
    "The final-test script refuses to run unless best_configs.json is tracked, unmodified and present in git history. This shows the configurations were fixed before the test split was scored. "
    "We do not call this pre-registration; it is a freeze in git history."))
def cfgs(k):
    c = cfg[k]
    if k.startswith("logreg"): return f"C = {c['C']}, l1_ratio = {c['l1_ratio']} (L2 penalty)"
    if k.startswith("rf"): return f"{c['n_estimators']} trees, max_depth = {c['max_depth']}, min_samples_leaf = {c['min_samples_leaf']}"
    return f"max_depth = {c['max_depth']}, learning_rate = {c['learning_rate']}, {c['n_estimators']} trees"
names = {"logreg": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}
rows = [["Model", "Features", "Chosen configuration (best_configs.json)"]]
for k in cfg:
    m, f = k.split("|")
    rows.append([names[m], f, cfgs(k)])
table(s, 0.6, 1.5, 7.9, rows, [2.1, 1.7, 4.1], size=13, row_h=0.46)
text(s, 0.6, 4.9, 7.9, 2.0, [
    ("One classifier per endpoint, labeled training rows only", {"bullet": True, "size": 16}),
    ("Imbalance: class weights (LR, RF), scale_pos_weight (XGBoost)", {"bullet": True, "size": 16}),
    ("Tuning: 8 configs per model and feature set, mean validation PR AUC, seed 0, scaffold split", {"bullet": True, "size": 16}),
], size=16)
fb = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.8), Inches(1.5), Inches(3.95), Inches(5.2))
fb.fill.solid(); fb.fill.fore_color.rgb = LIGHT; fb.line.color.rgb = RED
text(s, 8.95, 1.6, 3.7, 5.0, [
    ("Freeze proof", {"bold": True, "size": 20, "color": RED}),
    ("041892c  2026-10-25 19:41:39 +05:30\nfreeze tuned configs before test run", {"size": 15}),
    ("then", {"size": 15, "color": GREY}),
    ("043c2ed  2026-10-25 19:52:52 +05:30\nfinal test results: 432 rows and 36 prediction files", {"size": 15}),
    ("041892c is an ancestor of 043c2ed. The final-test script refuses to run unless best_configs.json is tracked, unmodified and in git history.", {"size": 15}),
], size=15)

# ---------- 6 evaluation protocol ----------
s = new_slide("Evaluation protocol", (
    "Metrics: ROC AUC and PR AUC per endpoint on labeled rows, then the mean over the 12 endpoints, then mean and sample standard deviation over seeds 0, 1 and 2. "
    "PR AUC is reported next to ROC AUC because positives are rare: a random model's PR AUC equals the positive rate, while ROC AUC can look good because negatives dominate. "
    "Scaffold split: Bemis-Murcko scaffolds, largest groups go to train; all 1,753 acyclic molecules are in train, so scaffold validation and test contain only ring systems. "
    "The seed controls model randomness only; the splits are fixed, so the seed standard deviation does not measure split-to-split variation. "
    "Logistic Regression is deterministic, so its standard deviation is 0.0000. The test split has between 20 (NR-AR-LBD) and 109 (SR-ARE) positives per endpoint. "
    "Final runs: 3 models x 2 feature sets x 2 splits x 3 seeds x 12 endpoints = 432 fitted classifiers."))
text(s, 0.6, 1.5, 12.1, 5.4, [
    ("Masked metrics: ROC AUC and PR AUC per endpoint on labeled rows; NaN if an evaluation set has a single class", {"bullet": True}),
    ("PR AUC is shown beside ROC AUC because positives are rare (a random model's PR AUC equals the positive rate)", {"bullet": True}),
    ("Scaffold split (Bemis–Murcko): largest scaffold groups go to train; all 1,753 acyclic molecules are in train, so scaffold validation and test hold only ring systems", {"bullet": True}),
    ("Random split also saved: 6,167 train / 771 validation / 771 test for both splits", {"bullet": True}),
    ("Seeds 0, 1, 2 change model randomness only; the splits are fixed", {"bullet": True}),
    ("432 final classifiers = 3 models × 2 feature sets × 2 splits × 3 seeds × 12 endpoints", {"bullet": True}),
    ("Scaffold test has 20 (NR-AR-LBD) to 109 (SR-ARE) positives per endpoint, so endpoint scores are noisy", {"bullet": True}),
], size=19)

# ---------- 7 results ----------
s = new_slide("Results: scaffold test split", (
    "Scaffold split test results, mean over 12 endpoints, then mean and standard deviation over seeds 0, 1, 2 (tables/results_summary_scaffold.csv, from results/results_test.csv). "
    "Random Forest on morgan+desc has the highest mean in our runs: ROC AUC 0.7762 and PR AUC 0.4035. XGBoost on morgan+desc is close: ROC 0.7606 and PR 0.4010. "
    "The PR AUC difference between them is about 0.003, smaller than the seed standard deviation of 0.004 to 0.0045, so on PR AUC we treat them as tied; on ROC AUC the Random Forest is higher by about 0.016. "
    "Seeds do not capture split variance, and we ran no significance test. Logistic Regression has standard deviation 0.0000 because it is deterministic. "
    "With Morgan only, XGBoost (ROC 0.7020) is below Logistic Regression (0.7156), so the tree models are not uniformly ahead."))
rows = [["Model", "Features", "ROC AUC", "PR AUC"]]
bold = set()
for i, r in summ.iterrows():
    rows.append([r.model, r.feature_set, f"{r.ROC_AUC_mean:.4f} ± {r.ROC_AUC_std:.4f}", f"{r.PR_AUC_mean:.4f} ± {r.PR_AUC_std:.4f}"])
ib = summ.ROC_AUC_mean.idxmax() + 1; ip = summ.PR_AUC_mean.idxmax() + 1
bold |= {(ib, 2), (ip, 3)}
table(s, 0.6, 1.5, 8.2, rows, [2.4, 1.9, 1.95, 1.95], size=15, row_h=0.5, bold_cells=bold)
text(s, 0.6, 5.2, 8.2, 1.6, [("Scaffold test; mean over 12 endpoints, then mean ± sample sd over seeds 0, 1, 2. Bold + teal highlight marks the column best. LR is deterministic (sd 0.0000).", {"size": 14, "color": GREY})])
text(s, 9.1, 1.5, 3.7, 5.3, [
    ("Reading the table", {"bold": True, "size": 19}),
    ("Random Forest + descriptors has the highest mean ROC (0.7762) and PR AUC (0.4035) in our runs", {"bullet": True, "size": 16}),
    ("XGBoost + descriptors is close: 0.7606 / 0.4010. The PR gap (about 0.003) is below the seed sd (0.004–0.0045)", {"bullet": True, "size": 16}),
    ("With Morgan only, XGBoost ROC (0.7020) is below Logistic Regression (0.7156)", {"bullet": True, "size": 16}),
], size=16)

# ---------- 8 descriptors ----------
s = new_slide("Do descriptors help? Paired gain from adding them", (
    "Gain from morgan to morgan+desc on the scaffold test split, paired by seed (tables/descriptor_benefit.csv). Logistic Regression: ROC +0.0162, PR +0.0006, so essentially no PR gain. "
    "Random Forest: ROC +0.0474 (sd 0.0074), PR +0.0333 (sd 0.0031). XGBoost: ROC +0.0586 (sd 0.0059), PR +0.0829 (sd 0.0025). "
    "The two tree models gain more from descriptors than Logistic Regression does. We did not run an ablation or feature-importance analysis on the hybrid set, so we do not claim why."))
rows = [["Model", "Δ ROC AUC (mean ± sd)", "Δ PR AUC (mean ± sd)"]]
for _, r in desc.iterrows():
    rows.append([r.model_label, f"{r.Delta_ROC_mean:+.4f} ± {r.Delta_ROC_std:.4f}", f"{r.Delta_PR_mean:+.4f} ± {r.Delta_PR_std:.4f}"])
table(s, 0.6, 1.6, 7.6, rows, [2.6, 2.5, 2.5], size=16, row_h=0.55)
text(s, 0.6, 4.0, 7.6, 2.8, [
    ("Logistic Regression: PR AUC gain is +0.0006, i.e. essentially none", {"bullet": True}),
    ("Random Forest and XGBoost gain more: PR +0.0333 and +0.0829", {"bullet": True}),
    ("Δ = morgan+desc minus morgan, scaffold test, paired by seed", {"bullet": True, "color": GREY}),
], size=18)
picture(s, ROOT / "figures/descriptor_benefit.png", 8.4, 1.6, 4.5, 4.8)

# ---------- 9 gap ----------
s = new_slide("Random vs scaffold split: the generalization gap", (
    "Same model and feature set (morgan+desc), random split versus scaffold split (tables/scaffold_penalty.csv). Random: ROC 0.8405 / 0.8607 / 0.8621 and PR 0.4818 / 0.5456 / 0.5647 for LR / RF / XGB. "
    "Scaffold: 0.7317 / 0.7762 / 0.7606 ROC and 0.3122 / 0.4035 / 0.4010 PR. Differences (scaffold minus random): ROC -0.1088 / -0.0845 / -0.1015, PR -0.1695 / -0.1421 / -0.1637. "
    "All three models score lower on the scaffold split. Caveat to say out loud: the random and scaffold test sets contain different molecules and different positive counts, and the scaffold test has no acyclic molecules, "
    "so this is a difference between two evaluations, not a controlled measurement of leakage. We ran no significance test."))
rows = [["Model", "Random\nROC", "Scaffold\nROC", "Δ ROC", "Random\nPR", "Scaffold\nPR", "Δ PR"]]
for _, r in pen.iterrows():
    rows.append([r.model_label, f4(r.Random_ROC), f4(r.Scaffold_ROC), f"{r.Delta_ROC_scaf_minus_rand:+.4f}", f4(r.Random_PR), f4(r.Scaffold_PR), f"{r.Delta_PR_scaf_minus_rand:+.4f}"])
table(s, 0.6, 1.5, 7.6, rows, [2.1, 0.9, 0.9, 0.85, 0.9, 0.9, 0.85], size=13, row_h=0.55)
text(s, 0.6, 3.9, 7.6, 3.0, [
    ("Morgan+Desc, mean over 12 endpoints and 3 seeds", {"size": 14, "color": GREY}),
    ("All three models score lower on the scaffold split: ROC drops 0.0845 to 0.1088, PR AUC drops 0.1421 to 0.1695", {"bullet": True, "size": 17}),
    ("Caveat: the two test sets contain different molecules and positive counts, so this is not a controlled leakage measurement. No significance test was run", {"bullet": True, "size": 17}),
], size=17)
picture(s, ROOT / "figures/scaffold_vs_random_split.png", 8.35, 1.6, 4.6, 3.0)

# ---------- 10 per-endpoint ----------
s = new_slide("Per-endpoint results (scaffold test, Morgan+Desc)", (
    "Per-endpoint means over seeds from tables/endpoint_results_scaffold.csv, three decimals as in the table. Bold marks the highest value in each row for that metric. "
    "Random Forest has the highest ROC AUC on 9 of 12 endpoints and XGBoost on 3; for PR AUC Random Forest leads on 8 and XGBoost on 4; Logistic Regression leads on none. "
    "The lowest best PR AUCs are SR-ATAD5 (0.247) and SR-HSE (0.267). Each endpoint's scaffold test set has only 20 to 109 positives, so single-endpoint differences are noisy and should not be over-read."))
piv_r = ep.pivot(index="endpoint", columns="model_label", values="ROC_mean")
piv_p = ep.pivot(index="endpoint", columns="model_label", values="PR_mean")
order = ["Logistic Regression", "Random Forest", "XGBoost"]
rows = [["Endpoint", "LR\nROC", "RF\nROC", "XGB\nROC", "LR\nPR", "RF\nPR", "XGB\nPR"]]
bold = set()
for i, e in enumerate(piv_r.index, start=1):
    vr = [piv_r.loc[e, m] for m in order]; vp = [piv_p.loc[e, m] for m in order]
    rows.append([e] + [f"{v:.3f}" for v in vr] + [f"{v:.3f}" for v in vp])
    bold.add((i, 1 + vr.index(max(vr)))); bold.add((i, 4 + vp.index(max(vp))))
table(s, 0.6, 1.4, 8.4, rows, [2.1, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05], size=12, row_h=0.4, bold_cells=bold)
text(s, 9.3, 1.5, 3.6, 5.3, [
    ("Bold + teal highlight = highest in the row", {"size": 14, "color": GREY}),
    ("RF has the highest ROC on 9 of 12 endpoints, XGBoost on 3", {"bullet": True, "size": 16}),
    ("PR AUC: RF highest on 8, XGBoost on 4, LR on none", {"bullet": True, "size": 16}),
    ("Lowest best PR AUC: SR-ATAD5 (0.247) and SR-HSE (0.267)", {"bullet": True, "size": 16}),
    ("Only 20 to 109 test positives per endpoint: single-endpoint differences are noisy", {"bullet": True, "size": 16}),
], size=16)

# ---------- 11 tanimoto ----------
s = new_slide("How novel is the scaffold test set? Tanimoto similarity", (
    "For each test molecule we computed the maximum Tanimoto similarity to any training molecule, using the 2,048-bit Morgan fingerprints and src/similarity.py. "
    "Mean maximum similarity: 0.428 for the scaffold split (median 0.396) and 0.584 for the random split (median 0.575). "
    "A training analog with Tanimoto 0.85 or higher exists for 13 of 771 scaffold test molecules and 72 of 771 random test molecules. "
    "So the scaffold test set is clearly less similar to training, but not free of close analogs. These values were recomputed from the committed splits and fingerprints. "
    "Figure: figures/tanimoto_similarity_distributions.png."))
picture(s, ROOT / "figures/tanimoto_similarity_distributions.png", 0.6, 1.5, 7.6, 5.3)
text(s, 8.5, 1.6, 4.4, 5.2, [
    ("Mean max Tanimoto, test → train", {"bold": True, "size": 19}),
    ("Scaffold split: 0.428", {"bullet": True, "size": 20}),
    ("Random split: 0.584", {"bullet": True, "size": 20}),
    ("Analog with T ≥ 0.85: 13 of 771 scaffold test molecules vs 72 of 771 random", {"bullet": True, "size": 17}),
    ("The scaffold test is less similar to training, but not free of close analogs", {"bullet": True, "size": 17}),
], size=17)

# ---------- 12 limitations ----------
s = new_slide("Limitations", (
    "Be honest about what the evidence does and does not show. One scaffold split and one random split, each at seed 0: the seed standard deviation covers model randomness only, so it understates uncertainty about which model is better. "
    "No significance tests. Test positives per endpoint 20 to 109. Random and scaffold test sets contain different molecules, so the gap is not a controlled measurement. "
    "Scaffold validation and test contain no acyclic molecules. Hyperparameters were tuned on the scaffold split and reused on the random split. "
    "Only fingerprints and 2D descriptors; we have not explained why descriptors help the tree models (no ablation or importance analysis on the hybrid set). "
    "Phase 2 will add a graph neural network."))
text(s, 0.6, 1.5, 12.1, 5.4, [
    ("One fixed scaffold split and one fixed random split: seed sd covers model randomness only, not split variation", {"bullet": True}),
    ("No significance tests; the RF–XGBoost PR AUC gap (about 0.003) is smaller than the seed sd", {"bullet": True}),
    ("Only 20 to 109 test positives per endpoint: endpoint-level results are noisy", {"bullet": True}),
    ("Random and scaffold test sets hold different molecules; the gap is not a controlled leakage measurement", {"bullet": True}),
    ("Scaffold validation and test contain no acyclic molecules (all 1,753 are in train)", {"bullet": True}),
    ("Hyperparameters tuned on the scaffold split and reused unchanged on the random split", {"bullet": True}),
    ("Fingerprints and 2D descriptors only; we did not test why descriptors help tree models", {"bullet": True}),
], size=19)

# ---------- 13 team and next ----------
s = new_slide("Team contributions, QA and next steps", (
    "Roles follow the Phase 1 team handbook and the README. Raghav: data cleaning, deduplication, EDA, splits, Logistic Regression. Ranajeet: Morgan fingerprints, RDKit descriptors, leak-free preprocessing, Random Forest. "
    "Parrv: repository structure and contract, evaluation harness, XGBoost, tuning and final-test scripts, reproducibility. Abhijeet: literature review, theory, results analysis, Tanimoto study, report. "
    "Malhar is Member 5. The team presentation lists README and slide polish, tests, and viva preparation; the README describes the role as QA and verification. The committed QA report says the 200-molecule smoke test and final-result reproduction were completed by Parrv, while the viva drill remains unrun. "
    "Known QA item: 01_eda.ipynb fails as committed because it expects a Google Colab path; the report proposes a portable-path fix. Phase 2 is a graph neural network evaluated on the same splits and protocol. "
    "Reproducibility: the QA report records that make final-test reproduced all 432 result rows to 1e-9."))
rows = [["Member", "Contribution"],
        ["Raghav", "Data cleaning and deduplication, EDA, scaffold and random splits, Logistic Regression"],
        ["Ranajeet", "Morgan fingerprints, RDKit descriptors, leak-free preprocessing, Random Forest"],
        ["Parrv", "Repository and contract, evaluation harness, XGBoost, tuning and final-test scripts"],
        ["Abhijeet", "Literature review, theory, results analysis, Tanimoto study, report"],
        ["Malhar (Member 5)", "README and slide polish, tests, QA communication, viva preparation"]]
table(s, 0.6, 1.5, 12.1, rows, [2.2, 9.9], size=16, row_h=0.55)
box(s, 0.6, 5.1, 3.85, 1.45, "QA RECORD\nSmoke test and 432-row reproduction documented by Parrv", fill=TEAL, size=15)
box(s, 4.75, 5.1, 3.85, 1.45, "MEMBER 5 FOCUS\nREADME and slides, verification notes, viva preparation", fill=NAVY, size=15)
box(s, 8.9, 5.1, 3.85, 1.45, "OPEN ITEMS\nViva drill pending; 01_eda notebook uses a Colab path", fill=RED, size=15)

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
polish_package(OUT)
print("saved", OUT, count[0], "slides")

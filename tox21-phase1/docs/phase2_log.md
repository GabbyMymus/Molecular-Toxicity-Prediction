# Phase 2 change log (Parrv)

## A1 - Create CLAUDE.md at git root - Fri Nov 13 19:46:10 IST 2026
- Goal: write CLAUDE.md with the project rules at the git root, do not commit, print it.
- Commands run:
ls CLAUDE.md tox21-phase1/docs/phase2_log.md
date
(Write tool) CLAUDE.md
mkdir -p tox21-phase1/docs
git status --short
git diff --stat
- Files created or changed: CLAUDE.md (new, untracked), tox21-phase1/docs/phase2_log.md (new). See git status output in this session.
- Commits: none
- Results: none (no numbers measured). Before the task, neither CLAUDE.md nor phase2_log.md existed (ls reported "No such file or directory").
- Problems and fixes: none.
- Open items: CLAUDE.md not committed, awaiting Parrv's decision. Log not committed.

## A1 fix - Replace Logging section in CLAUDE.md with full text - Fri Nov 13 19:47:36 IST 2026
- Goal: restore the full Logging text in CLAUDE.md, change nothing else.
- Commands run:
Edit CLAUDE.md (Logging section)
date
git status --short
git diff --stat
- Files created or changed: CLAUDE.md (untracked, Logging section edited); tox21-phase1/docs/phase2_log.md (this entry).
- Commits: none
- Results: none (no numbers measured).
- Problems and fixes: none.
- Open items: CLAUDE.md and the log remain uncommitted.

## C1 - Read-only audit of teammate branches - Fri Nov 13 19:48:06 IST 2026
- Goal: list files changed on teammate phase2 branches vs origin/parrv/dl-engine and flag forbidden paths.
- Commands run:
git fetch origin
git branch -r
git rev-parse --verify -q origin/<raghav|ranajeet|abhijeet|malhar>/phase2
git ls-tree -r --name-only origin/main | grep -i best_configs
git ls-tree -r --name-only origin/parrv/dl-engine | grep -i best_configs
- Files created or changed: only this log entry (tox21-phase1/docs/phase2_log.md). No checkout or merge.
- Commits: none
- Results: none of origin/raghav/phase2, origin/ranajeet/phase2, origin/abhijeet/phase2, origin/malhar/phase2 exist (git branch -r). Only best_configs.json (Phase 1) found on origin/main and origin/parrv/dl-engine; no best_configs_dl.json on either.
- Problems and fixes: none. git fetch printed nothing. A remote branch origin/Abhijeet exists (different name) and was not audited.
- Open items: confirm the real teammate branch names; origin/Abhijeet not audited.

## A1 cleanup - Check CLAUDE.md for duplicate Logging rules - Fri Nov 13 19:49:04 IST 2026
- Goal: remove the short Logging bullets if duplicated.
- Commands run:
grep -n -i 'logging\|log entry format\|Never write a number' CLAUDE.md
awk '/^- Logging:/{f=1} /^- Report back/{f=0} f' CLAUDE.md
- Files created or changed: only this log entry. CLAUDE.md unchanged.
- Commits: none
- Results: grep matched line 13 (Logging) and line 22 (Never write a number, inside the multi-line section) only. No duplicate found.
- Problems and fixes: none.
- Open items: none.

## F5 - Merge ranajeet/phase2 (ff) and origin/main into parrv/dl-engine - Sun Nov 15 02:21:01 IST 2026
- Goal: bring Ranajeet's tuning work and main into parrv/dl-engine, no push.
- Commands run:
git fetch origin
git checkout parrv/dl-engine
git pull
git diff --name-status origin/parrv/dl-engine...origin/ranajeet/phase2
git show origin/ranajeet/phase2:tox21-phase1/results/best_configs_dl_ranajeet.json
git merge --ff-only origin/ranajeet/phase2
python -m pytest tests/test_gate0_ranajeet.py tests/test_dl_smoke.py -q
git merge --no-commit --no-ff origin/main
git merge --abort
git merge --no-ff origin/main -m 'Merge origin/main into parrv/dl-engine'
python -m pytest tests -q
python -m pytest tests -v -p no:warnings
- Files created or changed: merge brought in Ranajeet's 2 files (best_configs_dl_ranajeet.json, test_gate0_ranajeet.py) and main's changes (README, report, figures, tables, tests/conftest.py, etc.). Notebook 01_eda.ipynb still has my earlier uncommitted modification.
- Commits: ff to e89410f (Ranajeet); merge commit 'Merge origin/main into parrv/dl-engine' (hash in git log below).
- Results: diff listed exactly 2 added files. JSON keys fpmlp|morgan and fpmlp|both only, no metric/test fields. Gate0+smoke: 19 passed, 235 warnings in 18.44s. Dry-run merge of origin/main: no conflicts. Full pytest tests -q: Fatal Python error: Segmentation fault, in tests/test_models.py::test_factory_returns_fit_predict_proba_and_tiny_fit_works[xgb], xgboost/data.py line 1502 _meta_from_numpy via set_label. 71% reached before crash; no final pass/fail summary.
- Problems and fixes: xgboost segfault (venv python3.14). Not fixed, per instructions. Not pushed.
- Open items: xgb segfault cause unverified; whether it also occurs before the merge is unverified.

## B2/B3 - Verify MPNN tuning outputs (commit pending OK) - Sun Nov 15 02:55:59 IST 2026
- Goal: check dl_tuning_val_parrv.csv and best_configs_dl_parrv.json before committing.
- Commands run:
ls -la results/ (inside tox21-phase1/)
find results -type f (git root)
cat results/best_configs_dl_parrv.json
cat results/dl_tuning_val_parrv.csv
python (pandas) check of configs, split, seed, per-config mean pr_auc, JSON match
git status --short
- Files created or changed: none by me. Both files were at the git root results/; moved with mv to tox21-phase1/results/ and the empty root results/ removed. Staged only those two files.
- Commits: b7fce45 MPNN tuning (validation only). Pushed to origin parrv/dl-engine (range 6276565..b7fce45, which also carries the F5 ff merge and main merge cb04cd1).
- Results (from tox21-phase1/results/dl_tuning_val_parrv.csv, 96 rows; moved there after the check): 8 configs (ids 0-7), split=scaffold only, seed=0 only, model=mpnn, feature_set=graph, 12 endpoints per config, 0 NaN metrics. Mean val PR AUC per config: 0=0.387534, 1=0.328855, 2=0.362917, 3=0.355482, 4=0.373704, 5=0.350610, 6=0.367458, 7=0.372346. Best = config 0. JSON has single key mpnn|none and equals config 0 params: True.
- Problems and fixes: files are in the wrong folder (git root results/ instead of tox21-phase1/results/). The CSV has no column naming the evaluation split, so "validation only" is inferred from the filename and columns (roc_auc, pr_auc, best_epoch), not proven by the file.
- Open items: validation-only still inferred from filename/columns, not proven by the file.

## B-gcn - GCN tuning (validation only) - Sun Nov 15 03:06:31 IST 2026
- Goal: run GCN tuning, rename outputs with _parrv_gcn suffix, verify.
- Commands run:
ls results/dl_tuning_val.csv results/best_configs_dl.json
python -m src.dl.run --mode tune --model gcn
mv results/dl_tuning_val.csv results/dl_tuning_val_parrv_gcn.csv
mv results/best_configs_dl.json results/best_configs_dl_parrv_gcn.json
python (pandas) check of configs, split, seed, per-config mean pr_auc, JSON match
git status --short
- Files created or changed: tox21-phase1/results/dl_tuning_val_parrv_gcn.csv and tox21-phase1/results/best_configs_dl_parrv_gcn.json (new, untracked). The run wrote the default names results/dl_tuning_val.csv and results/best_configs_dl.json; both were renamed, so no best_configs_dl.json remains.
- Commits: none
- Results (tox21-phase1/results/dl_tuning_val_parrv_gcn.csv, 96 rows): 8 configs, split=scaffold only, seed=0 only, model=gcn, 0 NaN. Mean val PR AUC per config: 0=0.378659, 1=0.405372, 2=0.362714, 3=0.389127, 4=0.376209, 5=0.391717, 6=0.373669, 7=0.379822. Best = config 1; JSON has only key gcn|none and equals config 1 params (hidden 256, layers 3, dropout 0.2, weight_decay 1e-05, lr 0.001): True.
- Problems and fixes: run.py writes the unsuffixed default filenames and would overwrite/merge into them; neither existed before the run (the earlier ls of results/ filtered on dl showed only the parrv/ranajeet files). Console output was redirected to /tmp/_x rather than the scratchpad.
- Open items: validation-only inferred from run.py tune code path and file columns. Outputs not committed.

## C2/C3 - Freeze DL configs - Sun Nov 15 03:09:20 IST 2026
- Goal: commit GCN tuning (step 1), merge tuning JSONs into best_configs_dl.json, commit as FREEZE, push.
- Commands run:
git add tox21-phase1/results/dl_tuning_val_parrv_gcn.csv tox21-phase1/results/best_configs_dl_parrv_gcn.json
git commit -m "GCN tuning (validation only)"
git push origin parrv/dl-engine
python (merge of results/best_configs_dl_*.json into results/best_configs_dl.json, aborts on conflicting key)
git add tox21-phase1/results/best_configs_dl.json
git commit -m "FREEZE: DL configs frozen before any test evaluation"
git push origin parrv/dl-engine
git rev-parse HEAD
git log --all --name-only --pretty=format: | grep (test/pred patterns)
- Files created or changed: committed dl_tuning_val_parrv_gcn.csv, best_configs_dl_parrv_gcn.json, best_configs_dl.json. Uncommitted: 01_eda.ipynb, CLAUDE.md, phase2_log.md, gate0.py, results/parts/.
- Commits: 5e768fe GCN tuning (validation only); 0aebd870dc2347971e37161ffbfc2b0f507c3b9f FREEZE: DL configs frozen before any test evaluation. Both pushed (b7fce45..5e768fe, 5e768fe..0aebd87).
- Results: best_configs_dl.json keys mpnn|none, gcn|none, fpmlp|morgan, fpmlp|both; no conflicting keys. History grep found no dl_results_test*, dl_ablation* or gcn/mpnn/fpmlp preds files. It did list Phase 1 files results/preds/{logreg,rf,xgb}_*.npy (36 files) and scripts/04_final_test.py, which are Phase 1.
- Problems and fixes: CLAUDE.md contains no freeze snippet; used own merge script.
- Open items: Phase 1 preds are test predictions from Phase 1, not DL; not inspected further.

## D2/D3 - Verify MPNN final test outputs (staged, commit pending OK) - Sun Nov 15 15:47:57 IST 2026
- Goal: verify results/dl_results_test_parrv_mpnn.csv, preds and curves, then stage the 13 files.
- Commands run:
git status --short
git merge-base --is-ancestor 0aebd87 HEAD
python (pandas/numpy/json checks of CSV, 6 preds, 6 curves)
git add results/dl_results_test_parrv_mpnn.csv results/preds/mpnn_graph_*_final.npy results/curves/mpnn_graph_*_final.json
git diff --cached --stat
- Files created or changed: 13 files staged (CSV, 6 npy, 6 json) under tox21-phase1/results/. Nothing else staged.
- Commits: none (waiting for OK). FREEZE 0aebd87 is an ancestor of HEAD: yes.
- Results: CSV 72 rows, model=mpnn, feature_set=graph, splits random+scaffold, seeds 0,1,2, 12 endpoints per run, no duplicates, 0 NaN in roc_auc and pr_auc. 6 preds each shape (771, 12) float32, 0 NaN. 6 curves valid JSON (lists of epoch records: epoch, train_loss, val_pr, val_roc). Mean over endpoints, then over seeds (std over seeds): random ROC 0.815486 (0.014998), PR 0.438777 (0.040678); scaffold ROC 0.728746 (0.013776), PR 0.306941 (0.018033). Source: tox21-phase1/results/dl_results_test_parrv_mpnn.csv.
- Problems and fixes: hyperparameters are not recorded in the CSV (columns: endpoint, roc_auc, pr_auc, n_test, n_pos_test, model, feature_set, split, seed, tag, variant, best_epoch) or in the curves, so config match with mpnn|none cannot be verified from the files.
- Open items: config match unverified; commit and push pending Parrv's OK.

## D2/D3 commit - MPNN finals committed and pushed - Sun Nov 15 15:50:12 IST 2026
- Goal: commit the 13 staged MPNN final files and push.
- Commands run:
git commit -m "MPNN finals (frozen config, 2 splits x 3 seeds)"
git push origin parrv/dl-engine
git rev-parse HEAD
- Files created or changed: the 13 files staged earlier. phase2_log.md left uncommitted.
- Commits: 54910cd04de5234bd3119a9958e85bf75a608b46 MPNN finals (frozen config, 2 splits x 3 seeds)
- Results: none new.
- Problems and fixes: none.
- Open items: hyperparameter match with mpnn|none still not verifiable from the files.

## ad hoc - Update main and create parrv/methods - Sun Nov 15 16:19:02 IST 2026
- Goal: checkout main, pull, create branch parrv/methods.
- Commands run:
git checkout main
git pull
git checkout -b parrv/methods
git fetch origin
git remote prune origin --dry-run
git ls-remote --heads origin
git checkout main
git merge --ff-only origin/main
git branch -d parrv/methods
git checkout -b parrv/methods
- Files created or changed: none in the working tree. Uncommitted files unchanged (01_eda.ipynb, CLAUDE.md, phase2_log.md, gate0.py, results/parts/ and others).
- Commits: none. main and parrv/methods both at 9553255 (Merge pull request #16 from GabbyMymus/raghav/phase2).
- Problems and fixes: git pull printed "error: some local refs could not be updated" (git fetch: cannot lock ref 'refs/remotes/origin/abhijeet/phase2': 'refs/remotes/origin/abhijeet' exists). Remote has both Abhijeet and abhijeet/phase2; they collide on this case-insensitive filesystem. prune dry-run listed nothing. Main was not fast-forwarded by the pull, so the first parrv/methods was created from stale main 10cd3b1; I fast-forwarded main with git merge --ff-only origin/main and deleted and recreated the empty parrv/methods from it.
- Open items: local tracking ref origin/abhijeet/phase2 cannot be created (case collision). git fetch origin keeps printing that error.

## F1 - Methods section (methods.tex) - Sun Nov 15 16:21:30 IST 2026
- Goal: write tox21-phase1/report_phase2/sections/methods.tex from the code and best_configs_dl.json.
- Commands run:
git branch --show-current
ls report_phase2 report_phase2/sections src/dl src
cat report_phase2/sections/hybrid.tex, src/dl/train.py, models.py, ensemble.py, run.py, graph.py (head), splits.py, evaluate.py (head), config.py
git log --all --diff-filter=A (results/best_configs_dl.json, dl_results_test*, preds/mpnn*, gcn*, fpmlp*)
git show --stat 0aebd87
grep for bib keys in report/references.bib
(Write tool) tox21-phase1/report_phase2/sections/methods.tex; python edit of 4 phrases
- Files created or changed: tox21-phase1/report_phase2/sections/methods.tex (new, untracked). No other file touched.
- Commits: none
- Results: no result numbers written in the section. Facts taken from code: DEFAULTS in train.py (batch 64, max_epochs 60, patience 10, grad_clip 5, pos_weight cap 50, ReduceLROnPlateau factor 0.5 patience 5); GRIDS in run.py (8 configs each for mpnn, gcn, fpmlp); frozen configs from results/best_configs_dl.json (keys mpnn|none, gcn|none, fpmlp|morgan, fpmlp|both). git log: 0aebd87 FREEZE added only best_configs_dl.json; DL test-result files first added in 8547142, 54910cd, 60dccb4, all after it.
- Problems and fixes: report_phase2/sections/dataset.tex does not exist (only hybrid.tex), so style was matched to hybrid.tex and report/sections/models_setup.tex. LaTeX not compiled: pdflatex and latexmk are not installed here.
- Open items: UNVERIFIED list given to Parrv in the reply (H2 selection rule, freeze-before-any-run, no bib keys for GCN/AdamW/Set2Set/GRU, descriptor dimension, LaTeX compile).

GCN finals config confirmed by Raghav as gcn|none from best_configs_dl.json, 16:25 Nov 15.

## F1 fixes - Citations, descriptor dimension, FP-MLP feature set; commit and push - Sun Nov 15 17:16:11 IST 2026
- Goal: add four bib entries and cites, state descriptor dimension and FP-MLP finals feature set, commit methods.tex and references.bib.
- Commands run:
python (np.load features/descriptors_raw.npy shape; load_features morgan+desc scaffold and random shapes; groupby of results/dl_results_test_ranajeet_fpmlp.csv)
cat >> report/references.bib
python edit of report_phase2/sections/methods.tex
git add tox21-phase1/report/references.bib tox21-phase1/report_phase2/sections/methods.tex
git diff --cached --stat
git commit -m "Methods section (Phase 2)"
git push -u origin parrv/methods
- Files created or changed: report/references.bib (+32 lines), report_phase2/sections/methods.tex (new, 52 lines). phase2_log.md left uncommitted.
- Commits: 0c2161f Methods section (Phase 2), pushed to origin parrv/methods.
- Results: features/descriptors_raw.npy shape (7709, 217). load_features("morgan+desc", split) gives 2261 columns for scaffold and random, i.e. 213 descriptor columns. dl_results_test_ranajeet_fpmlp.csv: only model fpmlp+both, feature_set both, splits scaffold and random, seeds 0,1,2, 12 endpoints each.
- Problems and fixes: expected 213 is the processed count; the raw file has 217 columns (the text states both). Cho et al. full author list, and first names for the other three entries, were filled from my own knowledge, not from the prompt. Cho EMNLP pages not added.
- Open items: LaTeX not compiled (pdflatex not installed). Bib author names and Set2Set title capitalization to be checked by Parrv. H2 selection rule left as "selected manually".

## ad hoc - Create parrv/makefile-readme from origin/main - Sun Nov 15 17:25:12 IST 2026
- Goal: fetch origin main and branch parrv/makefile-readme from it.
- Commands run:
git fetch origin main
git checkout -b parrv/makefile-readme origin/main
- Files created or changed: none. Uncommitted files carried over unchanged.
- Commits: none. Branch at 2159503.
- Results: none.
- Problems and fixes: none unless the command output above shows an error.
- Open items: branch not pushed.

## E - Makefile and README for Phase 2 (uncommitted, waiting for OK) - Sun Nov 15 17:26:42 IST 2026
- Goal: add Phase 2 Makefile targets and a Phase 2 README section on branch parrv/makefile-readme (created from origin/main 2159503; branch already existed and was checked out when I ran the create command).
- Commands run:
git fetch origin main
git checkout -b parrv/makefile-readme origin/main (failed: branch already exists; already on it at 2159503 per git reflog)
cat Makefile README.md README_PHASE2_STARTER.md requirements_phase2.txt .gitignore
ls results scripts tests features
python (pandas) summary of model/feature_set/tag/variant per results/dl_*test*.csv
grep for final/ablate/ensemble commands in repo
python edit of Makefile and README.md
make -n dl-guard | dl-tune | dl-final | dl-ablate | dl-ensemble | phase2-test
git diff --stat
- Files created or changed: tox21-phase1/Makefile (+51), tox21-phase1/README.md (+75). Nothing else edited.
- Commits: none
- Results: make -n printed the expected recipes for all six targets; no target was executed (no tuning, finals or tests run). Result CSV contents used for the README table: dl_results_test_parrv_mpnn.csv model mpnn; _raghav model gcn; _ranajeet_fpmlp model fpmlp+both; _ranajeet_h1 model mpnn+desc; _ranajeet_h2 model ens_xgb_mpnn+desc; ablation tags raghav dropedge/no_class_weight/no_dropout/no_mask/no_wd, ranajeet both/shuffled_desc/morgan.
- Problems and fixes: run.py tune mode writes/merges results/best_configs_dl.json, so dl-tune cannot avoid touching that file; added a refuse-if-exists guard. scripts/run_hybrid_analysis.py hardcodes /Users/ranajeetroy/... and is Ranajeet's, so no dl-analysis target was added. hybrid.tex does not say the H2 partner is chosen by validation; used xgb:morgan+desc with mpnn+desc:graph+desc as in hybrid.tex and the starter README.
- Open items: UNVERIFIED items listed in the reply to Parrv.

## E fixes - merge origin/main, dl-ensemble, README scope (uncommitted, waiting for OK) - Sun Nov 15 17:30:16 IST 2026
- Goal: merge origin/main (H2 fix c103398), update dl-ensemble to match H2 files, fix README scope note.
- Commands run:
git fetch origin main
git merge --no-ff origin/main -m "Merge origin/main into parrv/makefile-readme"
python (pandas) summary of results/dl_results_test_ranajeet_h2.csv and _h2_alt_mpnndesc.csv
git ls-files results/preds results/curves
python edit of Makefile and README.md
git diff --stat
make -n dl-ensemble
- Files created or changed: merge commit d3184a9 (Merge origin/main into parrv/makefile-readme, brought in c103398 and PR #19: hybrid.tex, H2 csvs, ens preds). Uncommitted edits: tox21-phase1/Makefile, tox21-phase1/README.md.
- Commits: d3184a9 merge commit (local only, not pushed).
- Results: h2.csv model ens_xgb_fpmlp+both, feature_set morgan+desc+both, 72 rows (random+scaffold, seeds 0,1,2); h2_alt_mpnndesc.csv model ens_xgb_mpnn+desc, feature_set morgan+desc+graph+desc, 72 rows. git ls-files lists files under results/preds and results/curves (tracked). make -n dl-ensemble printed the guard plus two ensemble commands; nothing executed.
- Problems and fixes: the FP-MLP partner flag is fpmlp+both:both, not fpmlp:both (ensemble.py loads {model}_{fs}_{split}_seed{k}_final.npy and the preds are named fpmlp+both_both_...). fetch printed no abhijeet ref error this time.
- Open items: flags for the two H2 runs inferred from output file names and ensemble.py, not recorded in the repo.

## E commit and PR - Makefile and README pushed, PR opened - Sun Nov 15 17:31:15 IST 2026
- Goal: add the ensemble-append sentence to README, commit Makefile and README only, push, open PR.
- Commands run:
python edit of README.md (one sentence)
git add tox21-phase1/Makefile tox21-phase1/README.md
git diff --cached --stat
git commit -m "Phase 2 Makefile targets and README section"
git push -u origin parrv/makefile-readme
gh pr create --base main --head parrv/makefile-readme --title "Phase 2 Makefile and README"
- Files created or changed: Makefile (+57 lines changed), README.md (+78 lines changed) committed. Not staged: 01_eda.ipynb, CLAUDE.md, gate0.py, results/parts/, phase2_log.md.
- Commits: 5b33f22 Phase 2 Makefile targets and README section (pushed).
- Results: none.
- Problems and fixes: none.
- Open items: PR link in the reply to Parrv.

## F1b - Methods: H2 selection rule (committed, PR opened) - Sun Nov 15 17:32:41 IST 2026
- Goal: replace "selected manually" in methods.tex H2 paragraph with the validation selection rule.
- Commands run:
git fetch origin main
git checkout -b parrv/methods-h2 origin/main
python (max val_pr per curve, mean over seeds 0-2, scaffold and random, for fpmlp+both, mpnn+desc, gcn, mpnn)
python edit of report_phase2/sections/methods.tex
git add tox21-phase1/report_phase2/sections/methods.tex
git commit -m "Methods: H2 selection rule"
git push -u origin parrv/methods-h2
gh pr create --base main --head parrv/methods-h2 --title "Methods: H2 selection rule"
- Files created or changed: report_phase2/sections/methods.tex (1 line).
- Commits: d128272 Methods: H2 selection rule (pushed).
- Results: recomputed mean peak val PR AUC over seeds 0-2 from results/curves/*_final.json on the scaffold split reproduces the ordering and the values quoted in hybrid.tex (fpmlp+both highest, then mpnn+desc, gcn, mpnn); on the random split the ordering is the same but values differ.
- Problems and fixes: hybrid.tex does not say "scaffold split" in the H2 paragraph; I added it after the recomputation above matched the hybrid.tex values on scaffold, not random.
- Open items: none.

## F2 - Phase 2 viva drill (uncommitted, branch parrv/viva-phase2) - Sun Nov 15 17:37:48 IST 2026
- Goal: write tox21-phase1/docs/viva_drill_phase2.md with 25 questions, short answers and evidence.
- Commands run:
git fetch origin main
git checkout -b parrv/viva-phase2 origin/main
cat docs/viva_drill.md, report_phase2/sections/hybrid.tex
python (pandas) mean/std over seeds per model, split and ablation tag from results/results_test.csv, dl_results_test*.csv, dl_ablation_test*.csv
python (compute_pos_weight on scaffold and random training labels)
python (per-endpoint win counts: H2 vs XGBoost, H1 vs FP-MLP)
git log dates for 0aebd87, 8547142, 54910cd, 60dccb4, f6346cb, 041892c
(Write tool) tox21-phase1/docs/viva_drill_phase2.md
- Files created or changed: tox21-phase1/docs/viva_drill_phase2.md (new, untracked). No other file touched.
- Commits: none
- Results: numbers in the file recomputed from results CSVs (scaffold PR AUC: RF m+d 0.4035, XGB m+d 0.4010, H2 0.3945, H1 0.3553, FP-MLP 0.3429, GCN 0.3397, MPNN 0.3069). Uncapped max pos_weight 38.35 (scaffold train), 34.57 (random train): cap 50 not binding. Win counts recomputed: H2 vs XGB scaffold 4 wins 8 losses; H1 vs FP-MLP scaffold 9 wins; H2 vs XGB random 9 wins.
- Problems and fixes: none.
- Open items: UNVERIFIED list at the end of the file (cap-50 rationale, PR-AUC-for-stopping rationale, pre-freeze local test runs, hybrid.tex p-values, original ablation commands).

## F2 commit and PR - viva drill pushed, PR opened - Sun Nov 15 17:38:30 IST 2026
- Goal: commit docs/viva_drill_phase2.md only, push, open PR.
- Commands run:
git add tox21-phase1/docs/viva_drill_phase2.md
git commit -m "Phase 2 viva drill"
git push -u origin parrv/viva-phase2
gh pr create --base main --head parrv/viva-phase2 --title "Phase 2 viva drill"
- Files created or changed: viva_drill_phase2.md committed. phase2_log.md and all other files left uncommitted.
- Commits: b5244e5 Phase 2 viva drill (pushed).
- Results: none.
- Problems and fixes: none.
- Open items: PR link in reply to Parrv.

## F3a - Report accuracy fixes (uncommitted, branch parrv/report-accuracy) - Sun Nov 15 17:41:10 IST 2026
- Goal: correct numbers and soften causal wording in hybrid.tex; add acyclic-molecule sentence to methods.tex.
- Commands run:
git fetch origin main
git checkout -b parrv/report-accuracy origin/main
python (pandas/scipy: mean and sample SD over seeds 0-2 for every number quoted in hybrid.tex; per-endpoint win/loss and Wilcoxon p; peak val PR from results/curves/*_final.json)
python (RDKit ring count on data/processed/tox21_clean.csv vs data/splits/scaffold_seed0.json)
python edit of report_phase2/sections/hybrid.tex and methods.tex
git diff --stat
- Files created or changed: report_phase2/sections/hybrid.tex (14 lines), report_phase2/sections/methods.tex (2 lines).
- Commits: none
- Results: numbers changed in hybrid.tex: FP-MLP ROC SD 0.0051->0.0054; H1 ROC SD 0.0066->0.0067; shuffled_desc PR SD 0.0025->0.0027, ROC SD 0.0049->0.0050; mpnn+morgan PR SD 0.0088->0.0093; mpnn+both PR SD 0.0089->0.0087; secondary-vs-primary win/loss 5/7 -> 7/5. All other quoted numbers matched. Wilcoxon p recomputed and matching: 0.1514, 0.0005 (4.88e-4 exact), 0.0093, 0.9697, 0.5186, 0.0923. RF morgan+desc scaffold PR 0.4035 > XGB 0.4010. RDKit: 7709 rows, 1753 acyclic, all 1753 in scaffold train, 0 in val, 0 in test.
- Problems and fixes: first RDKit attempt failed with "RingInfo not initialized"; fixed by calling Chem.GetSymmSSSR before GetRingInfo.
- Open items: oversmoothing motivation in hybrid.tex line 5 not verified or changed; "rigorous controls", "severe penalty" wording also softened.

## F3a commit and PR - report accuracy pushed, PR opened - Sun Nov 15 18:08:20 IST 2026
- Goal: commit hybrid.tex and methods.tex only, push, open PR.
- Commands run:
git add tox21-phase1/report_phase2/sections/hybrid.tex tox21-phase1/report_phase2/sections/methods.tex
git commit -m "Report accuracy fixes (SDs, wording, scaffold caveat)"
git push -u origin parrv/report-accuracy
gh pr create --base main --head parrv/report-accuracy --title "Report accuracy fixes (SDs, wording, scaffold caveat)"
- Files created or changed: hybrid.tex and methods.tex committed; everything else left uncommitted.
- Commits: a867b1a Report accuracy fixes (SDs, wording, scaffold caveat) (pushed).
- Results: none.
- Problems and fixes: none.
- Open items: PR link in reply to Parrv.

## F2b - Viva drill H2 win count (blocked: PR #22 not merged) - Sun Nov 15 18:09:36 IST 2026
- Goal: branch parrv/viva-fix from origin/main after PR #22 merges and fix the 5/7 mention in docs/viva_drill_phase2.md.
- Commands run:
git fetch origin main
git checkout -b parrv/viva-fix origin/main
grep -n "5/7\|12/0" tox21-phase1/docs/viva_drill_phase2.md
gh pr view 22 --json state,mergedAt,title
git checkout parrv/report-accuracy
git branch -d parrv/viva-fix
- Files created or changed: none. Empty local branch parrv/viva-fix created and deleted; back on parrv/report-accuracy.
- Commits: none
- Results: gh pr view 22: state OPEN, mergedAt null. docs/viva_drill_phase2.md is not on origin/main (grep: No such file or directory).
- Problems and fixes: PR #22 is still open, so the file does not exist on origin/main and the requested edit cannot be made from there.
- Open items: waiting for Parrv to merge PR #22 or to say to branch from parrv/viva-phase2 instead.

## R1 - Integrate Abhijeet's report branch (merge in progress, not committed) - Sun Nov 15 18:39:00 IST 2026
- Goal: merge abhijeet/phase2 into parrv/integrate-report from origin/main (942310d), resolve conflicts per Parrv's rules, check inputs and cites.
- Commands run:
git fetch origin main
git checkout -b parrv/integrate-report origin/main
git fetch origin refs/heads/abhijeet/phase2:refs/heads/abhijeet-phase2-local (printed "unable to update local ref" for origin/abhijeet/phase2; the local ref was created at 004f043)
git merge abhijeet-phase2-local --no-commit
git checkout --ours -- sections/dataset.tex methods.tex regularization.tex setup.tex; git add of those four
python (regex check of \input, \cite, \ref, \includegraphics in report_phase2/)
- Files created or changed: merge staged, not committed. Conflicts (add/add): dataset.tex, methods.tex, regularization.tex, setup.tex; all four resolved to origin/main's version. hybrid.tex had no conflict and equals origin/main's. His version of everything else kept (main.tex, intro, lit_review, results, discussion, conclusion, ablations, validation, references_dl.bib, figures, notebook 06, scripts, slides_phase2, .gitignore).
- Commits: none.
- Results: all 12 \input files in report_phase2/main.tex exist. 18 distinct cite keys used; 0 missing from both bibs; 3 keys exist only in report/references.bib (cho2014gru, loshchilov2019adamw, vinyals2016set2set, cited in methods.tex) while main.tex loads only references_dl. No undefined \ref targets. All includegraphics files found. TODO stubs on this branch: ablations.tex, validation.tex (setup.tex stub replaced by main's real file).
- Problems and fixes: setup.tex conflict was not covered by the rules (main has a 23-line real section, his is a 1-line TODO); kept main's.
- Open items: add the three keys to references_dl.bib or add report/references to \bibliography; PR #23 (hybrid.tex fixes) is still open, so hybrid.tex here lacks them; pdflatex not run.

## R1b - Bib fix, merge commit, merge origin/main (stopped before push) - Sun Nov 15 18:41:46 IST 2026
- Goal: add three bib entries, commit the Abhijeet merge, merge latest origin/main, confirm hybrid.tex SDs, push, open PR.
- Commands run:
python (check keys absent in references_dl.bib; append cho2014gru, loshchilov2019adamw, vinyals2016set2set verbatim from report/references.bib)
git add tox21-phase1/report_phase2/references_dl.bib
git commit -m "Integrate Abhijeet's Phase 2 report skeleton"
git fetch origin main
git merge origin/main
gh pr view 22|23|24|25 --json state,title
grep "0.7269" report_phase2/sections/hybrid.tex
- Files created or changed: references_dl.bib (+3 entries, no other entry changed). Merge commit created locally.
- Commits: 0c1220d Integrate Abhijeet's Phase 2 report skeleton (local, not pushed).
- Results: git merge origin/main: Already up to date (origin/main still 942310d). PR #22 OPEN, #23 OPEN, #24 MERGED, #25 MERGED. hybrid.tex contains FP-MLP ROC "0.7269 \pm 0.0051", i.e. the old SD, not 0.0054.
- Problems and fixes: PRs #22 and #23 are not merged, so the corrected SDs are not on origin/main. Stopped before pushing and opening the PR.
- Open items: waiting for Parrv to merge #23 (and #22) or to say to merge parrv/report-accuracy into this branch.

## R1c - Merge report-accuracy, push, open PR - Sun Nov 15 18:42:54 IST 2026
- Goal: merge origin/parrv/report-accuracy into parrv/integrate-report, push, open PR.
- Commands run:
git fetch origin parrv/report-accuracy
git merge origin/parrv/report-accuracy
grep checks of hybrid.tex and methods.tex
git push -u origin parrv/integrate-report
gh pr create --base main --head parrv/integrate-report --title "Integrate Phase 2 report"
- Files created or changed: merge changed hybrid.tex and methods.tex (8 insertions, 8 deletions), no conflicts.
- Commits: 0b8b60b merge commit (pushed); 0c1220d Integrate Abhijeet's Phase 2 report skeleton.
- Results: hybrid.tex now has FP-MLP ROC 0.7269 pm 0.0054; methods.tex contains the 1,753 acyclic molecules sentence; no conflict markers in report_phase2/sections.
- Problems and fixes: none.
- Open items: PR link in reply to Parrv. pdflatex not run.

## ad hoc - Create parrv/results-fix from origin/main - Sun Nov 15 22:26:31 IST 2026
- Goal: fetch origin main and branch parrv/results-fix.
- Commands run:
git fetch origin main
git checkout -b parrv/results-fix origin/main
- Files created or changed: none. Uncommitted files carried over unchanged.
- Commits: none. Branch at 6e9392c.
- Results: none.
- Problems and fixes: none.
- Open items: none.

## R2a - Results section, table script (uncommitted, branch parrv/results-fix) - Sun Nov 15 22:31:56 IST 2026
- Goal: generate main results table from CSVs, rewrite results.tex claims that conflict with the data and hybrid.tex.
- Commands run:
git fetch origin main; git checkout -b parrv/results-fix origin/main (done in the previous turn)
python (pandas/scipy/RDKit-free recomputation: table values, per-endpoint stats, Wilcoxon p, gaps, bootstrap CIs with src.dl.analysis.paired_bootstrap_ci, curves for seed 0)
(Write tool) tox21-phase1/scripts/make_results_table.py
python scripts/make_results_table.py
python edit of report_phase2/sections/results.tex
git diff --stat
- Files created or changed: scripts/make_results_table.py (new), report_phase2/tables/main_results.tex (new, generated), report_phase2/sections/results.tex (edited).
- Commits: none
- Results: see the change list in the reply to Parrv. Old hand-typed table matched the CSVs except H2-secondary ROC SD 0.0017 -> 0.0016.
- Problems and fixes: figure mean_pr_roc_scaffold shows H2 = XGB+H1 and omits GCN, MPNN, primary H2 (stale); not regenerated, caption now says so. LaTeX not compiled (pdflatex missing), so table fit is by design (single column, 3 columns), not measured.
- Open items: other figures (heatmap, generalization gap, ablation, learning curves) not checked against the current CSVs.

## R2a commit and PR - results section committed and pushed - Sun Nov 15 22:35:02 IST 2026
- Goal: commit results.tex, main_results.tex, make_results_table.py and phase2_log.md; push; open PR.
- Commands run:
git add tox21-phase1/report_phase2/sections/results.tex tox21-phase1/report_phase2/tables/main_results.tex tox21-phase1/scripts/make_results_table.py tox21-phase1/docs/phase2_log.md
git commit -m "Report: rewrite results section from committed CSVs (add GCN, new H2, drop overclaims)"
git push -u origin parrv/results-fix
gh pr create --base main --head parrv/results-fix
git status --short
- Files created or changed: the four staged files. Not staged: CLAUDE.md, gate0.py, results/parts/, notebooks/01_eda.ipynb.
- Commits: see git log (hash and PR URL in the reply to Parrv).
- Results: none new.
- Problems and fixes: none.
- Open items: stale figure mean_pr_roc_scaffold; other figures unchecked; LaTeX not compiled.

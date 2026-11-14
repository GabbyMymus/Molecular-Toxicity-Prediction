# QA and verification report (Member 5 role, handbook section 12)

Date: 2026-10-26. Base: `origin/main` `1cdd877`. All checks were run in fresh clones outside the working repository.
Every statement is marked VERIFIED (observed in this run) or CANNOT VERIFY. The QA tasks below were run in fresh clones outside the working repository.

## 1. Smoke test (`tests/test_smoke.py`)
- 200 molecules (seeded sample of the committed cleaned CSV, read only) go through clean, scaffold split, Morgan and
  descriptor features, the train-only descriptor preprocessor, Logistic Regression (first grid entry) and per-endpoint scoring.
- Checks: row order kept, missing labels never filled (NaN count identical before and after), splits disjoint and complete,
  descriptors finite, probabilities in [0, 1], metrics in [0, 1], and nothing written under `results/` or `data/splits/`
  (file sizes and mtimes compared before and after). The real test split is never read.
- Marked `smoke`. Run with `python -m pytest -m smoke`. VERIFIED: passes; 40 s on the first (cold) run, 1.3 s for the test body when warm (2.5 s total).

## 2. Notebooks, run top to bottom on a clean kernel (`jupyter nbconvert --execute` into a scratch folder; committed notebooks untouched)
| Notebook | Result |
|---|---|
| `02_features.ipynb` | VERIFIED: runs, 0 errors, text outputs identical to the committed ones. |
| `03_analysis.ipynb` | VERIFIED: runs, 0 errors, text outputs identical except the printed `ROOT` path. It also overwrites 5 committed figures (PNG bytes differ). |
| `01_eda.ipynb` | **FAILS as committed.** Cell 2 reads `/content/tox21.csv` (a Google Colab path) and fails with FileNotFoundError. It also runs `%pip install` and writes its cleaned data, splits and features under `/content/tox21_project`. |

- `01_eda.ipynb` after redirecting the two Colab paths in a scratch copy (raw file to `../data/raw/tox21.csv.gz`, output to a
  scratch folder): VERIFIED: runs, 0 errors. Text output differs from the committed output only in RDKit log timestamps,
  saved-path strings and a library path in one warning. Seven code cells (1, 7, 13, 17, 18, 20, 29) differ in text. A
  token-level comparison after removing timestamps shows: cells 7 and 17 identical; cell 1 differs only in pandas'
  dtype names (`object` vs `str`, pandas 3.1); cell 13 differs in how many repeated RDKit log lines were kept (the committed
  output was truncated by Colab); cells 18, 20 and 29 differ only in paths. No numeric difference found.
- Suggested fix (owner Raghav): set `RAW_PATH = Path("../data/raw/tox21.csv.gz")`, point `OUT_DIR` at a scratch folder that
  is not the committed `data/` or `features/`, and drop the `%pip install` cell. Not applied here (outside tests/docs).
- Absolute paths in tracked notebooks: `01_eda.ipynb` (`/content/...`) and `03_analysis.ipynb` source and saved output
  (`/Users/abhijeetkumar/...`, used as fallbacks; the notebook still runs elsewhere). This breaks the "no absolute paths" contract (owners: Raghav, Abhijeet).

## 3. Fresh clone following only the README
Python 3.14.7 (macOS, libomp installed), `python3 -m venv .venv`.
| Step | Result |
|---|---|
| `make setup` | VERIFIED: succeeds (23 s). |
| `make test` before `make features` | VERIFIED: 63 passed, 2 failed (`test_load_features_morgan`, `test_load_features_morgan_plus_desc`: FileNotFoundError, feature arrays not built). The README run table already says "Run `make features` first; two tests read the built feature arrays". |
| `PYTHONPATH=. python scripts/01_clean_and_split.py --download` | VERIFIED: raw 7,831 rows, clean 7,709, splits 6,167/771/771; `git status` showed no tracked file changed. |
| `make features` | VERIFIED: about 40 s; writes `morgan_2048.npy`, `descriptors_raw.npy`, `descriptor_names.txt`, `feature_manifest.json`; `git status` clean (manifest unchanged). |
| `make test` after `make features` | VERIFIED: 65 passed in 7.7 s. |
| `make final-test` (10 min 25 s) | VERIFIED: 432 rows; keys identical to the committed file; roc_auc, pr_auc, n_test, n_pos_test 0 differing rows at 1e-9; 36 prediction files (771, 12), max abs diff 2.2e-16, 12 RF files not bit-identical. Compared by key, the metric values are identical (max abs difference 0.0). The fresh file's row order differs from the committed file, so git shows `results_test.csv` as modified; 12 RF prediction files also differ at the 1e-16 level. |
| `make tune` | Not run (not needed). |
| `make eda` | VERIFIED: runs in 5 s; rewrites 4 committed `eda_*.png` files (binary differences only). |

## 4. `descriptor_names.txt`
VERIFIED: `make features` writes `features/descriptor_names.txt` with 217 names (first: `MaxAbsEStateIndex`). It is git-ignored, so it is not in a fresh clone until `make features` runs, as intended.

## 5. Repo hygiene: LICENSE and notes
- LICENSE: the handbook does not require one. Its only mention is the Member 5 bullet "check the README, .gitignore, requirements, licence/notes" (a check, not a deliverable), and its repository layout lists no LICENSE or notes file. Choosing a licence is a team decision (it affects who may reuse the code), so none was added.
- Notes file: not specified by the handbook. The repo has `README.md` and `CONTRACT.md`. This `docs/` folder holds QA notes.
- The course instructions PDF itself was not available to me; I only know what the handbook quotes from it. CANNOT VERIFY whether it asks for a licence.
- `.gitignore`, `requirements.txt`: present and used by the fresh clone above. `requirements.txt` has lower bounds only, no pins (CANNOT VERIFY future installs; the run above used scikit-learn 1.9.1, numpy 2.5.4, xgboost 3.4.1, rdkit 2026.9.1).

## 6. Not done or not verifiable
- Viva quiz with each teammate (handbook section 16): not run. `docs/viva_drill.md` maps each question to repo evidence.
- Python 3.11 and 3.12: not tested. The Homebrew Python 3.12 here fails at `ensurepip` when creating a venv (environment problem, not the repo); a uv-built 3.12 environment ran the smoke test.
- Bootstrap confidence intervals / error-by-scaffold analysis (suggested Member 5 technical piece): not done.

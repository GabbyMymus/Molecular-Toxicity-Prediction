"""Tests for scripts/03_tune.py, 04_final_test.py and merge_results.py (synthetic data, throwaway git repo)."""

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src import evaluate
from src.config import ENDPOINTS, FEATURE_SETS, SEEDS, SPLIT_NAMES

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
FLAG = "--i-know-this-touches-test"


def load_script(filename):
    spec = importlib.util.spec_from_file_location("script_" + Path(filename).stem, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tune_script = load_script("03_tune.py")
final_script = load_script("04_final_test.py")
merge_script = load_script("merge_results.py")


def make_dummy(params, y_train, seed):
    return DummyClassifier(strategy="prior")


DUMMY_REGISTRY = {name: (make_dummy, [{"a": 1}, {"a": 2}]) for name in ("logreg", "rf", "xgb")}
N_KEYS = len(DUMMY_REGISTRY) * len(FEATURE_SETS)


def git(repo, *args):
    cmd = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args]
    return subprocess.run(cmd, cwd=repo, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway git repo whose results/ dir replaces evaluate.RESULTS, with synthetic cached data."""
    git(tmp_path, "init", "-q")
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(evaluate, "RESULTS", tmp_path / "results")
    rng = np.random.default_rng(0)
    cache = {}
    for i, (fs, split) in enumerate((fs, sp) for fs in FEATURE_SETS for sp in SPLIT_NAMES):
        def ys(n):
            return (rng.random((n, len(ENDPOINTS))) < 0.2).astype(float)
        cache[(fs, split)] = (rng.normal(size=(120, 4)), rng.normal(size=(40, 4)), rng.normal(size=(40, 4)),
                              ys(120), ys(40), ys(40))
    monkeypatch.setattr(evaluate, "_DATA_CACHE", cache)
    return tmp_path


def freeze(repo):
    git(repo, "add", "results/best_configs.json")
    git(repo, "commit", "-q", "-m", "freeze best configs")


def tuned(repo):
    tune_script.main([], DUMMY_REGISTRY)
    return repo / "results" / "best_configs.json"


def test_tune_writes_six_keys_and_logs_tuning_rows(repo):
    best = json.loads(tuned(repo).read_text())
    assert sorted(best) == sorted(f"{m}|{fs}" for m in DUMMY_REGISTRY for fs in FEATURE_SETS)
    tv = pd.read_csv(repo / "results" / "tuning_val.csv")
    assert set(tv["split"]) == {"scaffold"} and set(tv["seed"]) == {0}


def test_tune_models_flag_merges_instead_of_overwriting(repo):
    path = tuned(repo)
    before = json.loads(path.read_text())
    before["xgb|morgan"] = {"a": 2}  # pretend an earlier run chose config 1
    path.write_text(json.dumps(before))
    tune_script.main(["--models", "rf"], DUMMY_REGISTRY)
    after = json.loads(path.read_text())
    assert after["xgb|morgan"] == {"a": 2} and len(after) == N_KEYS


def test_tune_models_flag_rejects_unknown_model(repo):
    with pytest.raises(SystemExit):
        tune_script.main(["--models", "svm"], DUMMY_REGISTRY)


def test_final_refuses_without_flag(repo):
    freeze_path = tuned(repo)
    assert freeze_path.exists()
    freeze(repo)
    with pytest.raises(SystemExit, match="i-know-this-touches-test"):
        final_script.main([], DUMMY_REGISTRY)
    assert not (repo / "results" / "results_test.csv").exists()


def test_final_refuses_when_best_configs_missing(repo):
    with pytest.raises(SystemExit, match="missing"):
        final_script.main([FLAG], DUMMY_REGISTRY)


def test_final_refuses_when_a_key_is_missing(repo):
    path = tuned(repo)
    best = json.loads(path.read_text())
    del best["xgb|morgan+desc"]
    path.write_text(json.dumps(best))
    freeze(repo)
    with pytest.raises(SystemExit, match="lacks 1 of 6 keys"):
        final_script.main([FLAG], DUMMY_REGISTRY)


def test_final_refuses_config_not_in_grid(repo):
    path = tuned(repo)
    best = json.loads(path.read_text())
    best["rf|morgan"] = {"a": 99}
    path.write_text(json.dumps(best))
    freeze(repo)
    with pytest.raises(SystemExit, match="not an entry"):
        final_script.main([FLAG], DUMMY_REGISTRY)


def test_final_refuses_untracked_best_configs(repo):
    tuned(repo)
    with pytest.raises(SystemExit, match="untracked or has uncommitted"):
        final_script.main([FLAG], DUMMY_REGISTRY)
    assert not (repo / "results" / "results_test.csv").exists()


def test_final_refuses_staged_but_uncommitted(repo):
    tuned(repo)
    git(repo, "add", "results/best_configs.json")
    with pytest.raises(SystemExit, match="untracked or has uncommitted"):
        final_script.main([FLAG], DUMMY_REGISTRY)


def test_final_refuses_modified_after_commit(repo):
    path = tuned(repo)
    freeze(repo)
    best = json.loads(path.read_text())
    best["rf|morgan"] = {"a": 2}
    path.write_text(json.dumps(best))
    with pytest.raises(SystemExit, match="uncommitted"):
        final_script.main([FLAG], DUMMY_REGISTRY)


def test_final_refuses_when_ignored_so_never_committed(repo):
    tuned(repo)
    (repo / ".gitignore").write_text("results/best_configs.json\n")
    with pytest.raises(SystemExit, match="no commit"):
        final_script.main([FLAG], DUMMY_REGISTRY)


def test_final_runs_when_frozen_and_overwrites_output(repo):
    tuned(repo)
    freeze(repo)
    out = final_script.main([FLAG], DUMMY_REGISTRY)
    expected = len(DUMMY_REGISTRY) * len(FEATURE_SETS) * len(SPLIT_NAMES) * len(SEEDS) * len(ENDPOINTS)
    assert len(out) == expected
    assert list(out.columns) == evaluate.RESULT_COLUMNS
    assert set(out["seed"]) == set(SEEDS) and set(out["split"]) == set(SPLIT_NAMES)
    csv = repo / "results" / "results_test.csv"
    assert len(pd.read_csv(csv)) == expected
    final_script.main([FLAG], DUMMY_REGISTRY)  # second run overwrites, never appends
    assert len(pd.read_csv(csv)) == expected
    assert len(list((repo / "results" / "preds").glob("*.npy"))) == expected // len(ENDPOINTS)


def test_parallel_parts_merge_to_the_full_result(repo):
    tuned(repo)
    freeze(repo)
    parts = repo / "results" / "parts"
    final_script.main([FLAG, "--models", "logreg,rf", "--out", str(parts / "a.csv")], DUMMY_REGISTRY)
    final_script.main([FLAG, "--models", "xgb", "--out", str(parts / "b.csv")], DUMMY_REGISTRY)
    assert not (repo / "results" / "results_test.csv").exists()
    merged = merge_script.main([])
    full = final_script.main([FLAG], DUMMY_REGISTRY)
    assert len(merged) == len(full)
    key = ["model", "feature_set", "split", "seed", "endpoint"]
    pd.testing.assert_frame_equal(
        merged.sort_values(key).reset_index(drop=True)[evaluate.RESULT_COLUMNS],
        full.sort_values(key).reset_index(drop=True)[evaluate.RESULT_COLUMNS],
        check_dtype=False,
    )


def test_merge_rejects_duplicate_rows(repo):
    tuned(repo)
    freeze(repo)
    p = repo / "results" / "parts" / "a.csv"
    final_script.main([FLAG, "--models", "rf", "--out", str(p)], DUMMY_REGISTRY)
    with pytest.raises(SystemExit, match="Duplicate"):
        merge_script.main([str(p), str(p)])


def test_concurrent_record_best_keeps_every_key(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "best_configs.json"
    keys = [f"{m}|{fs}" for m in ("logreg", "rf", "xgb") for fs in FEATURE_SETS]
    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        list(pool.map(lambda k: tune_script.record_best(path, k, {"a": 1}, merge=True, local={}), keys * 5))
    assert sorted(json.loads(path.read_text())) == sorted(keys)


def test_concurrent_tuning_rows_are_not_lost(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "tuning_val.csv"

    def write(model):
        for _ in range(5):
            frame = pd.DataFrame({"model": [model] * 3, "feature_set": ["morgan"] * 3, "split": ["scaffold"] * 3,
                                  "seed": [0] * 3, "config_id": [0, 1, 2]})
            evaluate._replace_rows(frame, path, ["model", "feature_set", "split", "seed"])

    models = ["a", "b", "c", "d"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, models))
    out = pd.read_csv(path)
    assert sorted(out["model"].unique()) == models and len(out) == 12

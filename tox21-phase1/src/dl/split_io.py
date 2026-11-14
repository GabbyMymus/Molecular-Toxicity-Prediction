"""Bridge: Phase 1 load_split takes a path; the DL code addresses splits by name."""
from src.config import SPLITS
from src.splits import load_split


def load_split_named(name, seed=0):
    return load_split(SPLITS / f"{name}_seed{seed}.json")

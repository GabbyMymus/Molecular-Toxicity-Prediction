"""Build Morgan fingerprints, raw RDKit descriptors, and feature manifest."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_clean
from src.features import build_descriptors, build_feature_manifest, build_morgan

print("Loading cleaned dataset...")
frame, _ = load_clean()
print(f"Loaded {len(frame)} molecules.")

print("Generating 2048-bit Morgan fingerprints (radius=2)...")
morgan = build_morgan(frame["smiles"])
print(f"Morgan fingerprints generated: shape {morgan.shape}, dtype {morgan.dtype}")

print("Generating 2D RDKit molecular descriptors...")
descriptors, names = build_descriptors(frame["smiles"])
print(f"Descriptors generated: shape {descriptors.shape}, dtype {descriptors.dtype}, {len(names)} features")

print("Generating feature manifest...")
manifest = build_feature_manifest()
print(f"Feature manifest generated with {manifest['descriptors']['n_raw_descriptors']} descriptors.")
print("Feature generation complete!")

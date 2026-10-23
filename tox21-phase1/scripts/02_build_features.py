from src.data import load_clean
from src.features import build_descriptors, build_morgan

frame, _ = load_clean()
build_morgan(frame["smiles"])
build_descriptors(frame["smiles"])


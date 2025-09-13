import sys
import pathlib

# Ensure ogame_env package is importable when tests run from repo root
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

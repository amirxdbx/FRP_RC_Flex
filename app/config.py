from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PACKAGE_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_pinn_phys.pth"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"


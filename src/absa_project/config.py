from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

LABELS = ["negative", "neutral", "positive", "conflict"]
ID2LABEL = dict(enumerate(LABELS))

DEFAULT_HF_MODEL_REPO = "praiseprince/absa-customer-insights-model"

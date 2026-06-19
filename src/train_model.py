from pathlib import Path
from src.data import load_transactions
from src.features import (
    compute_account_features,
    aggregate_labels_to_account,
)
from src.models import (
    train_models,
    save_artifacts,
)

TRAIN_DATA = "datasets/HI-Small_Trans.csv"

tx = load_transactions(TRAIN_DATA)
feat = compute_account_features(tx)
labels = aggregate_labels_to_account(tx)

artifacts = train_models(
    feat, 
    labels,
)

save_artifacts(
    artifacts,
    "artifacts/aml_model.pkl"
)
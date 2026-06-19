from src.data import load_transactions
from src.features import (
    compute_account_features,
    aggregate_labels_to_account,
)
from src.models import (
    load_artifacts,
    score_accounts,
    evaluate_if_labels_available,
)

TEST_DATA = "datasets/LI-Small_Trans.csv"

artifacts = load_artifacts("artifacts/aml_model.pkl")

tx = load_transactions(TEST_DATA)

feat = compute_account_features(tx)

labels = aggregate_labels_to_account(tx)

scored = score_accounts(
    feat, 
    artifacts,
)

metrics = evaluate_if_labels_available(
    scored, 
    labels,
)

print(metrics)
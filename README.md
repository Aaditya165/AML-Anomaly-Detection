# AML Transaction Graph Intelligence Dashboard

A Streamlit-based AML investigation prototype that uses only the transactions CSV to train and score accounts.

## What it does

- Loads one IBM AML `*_trans.csv` file
- Builds a directed transaction graph
- Engineers account-level graph and temporal features
- Trains:
  - a supervised classifier when `Is Laundering` is present
  - an unsupervised anomaly detector
- Produces an account risk score
- Shows:
  - high-risk graph view
  - alert feed
  - portfolio analytics
- Generates SHAP-style explanations when available

## Where the model lives

The model is defined and trained in `src/models.py`.

- `train_models(...)` builds the supervised and anomaly models
- `save_artifacts(...)` writes the trained `ModelArtifacts` object to a `.pkl` file
- `load_artifacts(...)` reads the `.pkl` file back into the app

`src/pipeline.py` orchestrates the end-to-end training flow. `app.py` is only the UI.

## Pickle workflow

Train in Colab or locally, then save the trained artifacts:

```python
from src.pipeline import run_pipeline
from src.models import save_artifacts

result = run_pipeline("/content/LI-Small_Trans.csv")
save_artifacts(result["artifacts"], "/content/aml_model.pkl")
```

Then copy `aml_model.pkl` to your local machine and load it in the app.

Important: pickle files should only be loaded from sources you trust.

## Project structure

- `app.py` — Streamlit UI
- `src/data.py` — loading and normalization
- `src/features.py` — feature engineering
- `src/models.py` — training, scoring, evaluation, pickle save/load
- `src/graph_viz.py` — Plotly graph rendering
- `src/explain.py` — explanations
- `src/pipeline.py` — orchestration

## Input data

Point the app at a directory containing one or more transaction CSV files. The expected columns are:

- `Timestamp`
- `From Bank`
- `Account`
- `To Bank`
- `Account` (duplicate name in raw IBM files is handled)
- `Amount Received`
- `Receiving Currency`
- `Amount Paid`
- `Payment Currency`
- `Payment Format`
- optional `Is Laundering`

Only the transactions file is required.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- If `Is Laundering` is missing, the supervised model is skipped and the system falls back to anomaly scoring.
- For large graphs, only the highest-risk subgraph is rendered.
- The code is designed as a prototype for internal AML review, not as a production alerting system.

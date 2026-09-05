# Sensor Fault Detection

A reproducible, local-first binary classification service for APS sensor fault prediction. It models the APS failure class (`class=1`) from numeric sensor signals. It is a portfolio implementation, not a maintenance-control system.

## Design

`CSV → validation → stratified train/validation/test split → median imputation + robust scaling → SMOTETomek (train only) → model comparison → validation threshold → untouched test evaluation → versioned artifact → FastAPI`.

Median imputation is used because zero can be a meaningful sensor reading. Models are a logistic-regression baseline and a random forest; validation PR-AUC selects the candidate. F1, precision, recall, ROC-AUC, PR-AUC and a confusion matrix are calculated only in a reproducible training run and saved in artifact metadata—no metrics are claimed here.

Data drift is a numeric-feature KS test with Benjamini-Hochberg FDR correction. Drift is distribution change, not evidence of model degradation.

## Quick start

```bash
python -m venv .venv
.venv\\Scripts\\pip install -r requirements.txt
.venv\\Scripts\\python -m sensor.cli train --source csv --path path\\to\\dataset.csv
.venv\\Scripts\\uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The CSV must have a binary `class` target and numeric feature columns. The command creates `artifacts/production/model.joblib` and metadata, including version, data hash, split sizes, threshold, hyperparameters and actual test metrics. Test data is never resampled, used to fit transformations, select a model, or choose a threshold.

Endpoints: `GET /health`, `GET /model/info`, `POST /predict`. A prediction body is `{ "features": { "aa_000": 1.2 } }`; it must contain exactly the features recorded in the trusted server-side artifact. There is deliberately no training endpoint.

## Configuration and deployment

Copy `.env.example` locally; never commit `.env`. MongoDB variables are placeholders and the app does not require MongoDB or AWS. S3 sync is disabled with `ENABLE_S3_SYNC=false`. The image deliberately does not bundle a serialized model; mount a trusted, trained artifact at runtime and set `MODEL_PATH`:

```bash
docker build -t sensor-fault-detection .
docker run -p 8000:8000 -v "${PWD}/artifacts/production:/models:ro" -e MODEL_PATH=/models/model.joblib sensor-fault-detection
```

CI installs, lints, tests, and builds the image. `GET /favicon.svg` serves the API favicon. A deployment needs a trusted trained artifact supplied through the deployment platform's private storage or volume; never accept artifact paths from API clients.

## Testing and limitations

```bash
pytest -q
```

Tests use synthetic local data only. The current model comparison is intentionally bounded; future work includes calibrated probabilities, approved feature schema versioning, monitoring alerting, and an authenticated model registry.

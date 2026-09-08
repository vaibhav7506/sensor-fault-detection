# Synthetic demo

`demo_sensor_data.csv` is a small, deterministic synthetic dataset. It exists only to exercise the CSV training and prediction flow; it is not representative of APS equipment and must never be used as a production model source.

Train a disposable local artifact:

```bash
python -m sensor.cli train --source csv --path examples/demo_sensor_data.csv --output-dir .demo-artifacts
```

Then start the API against that artifact:

```bash
MODEL_PATH=.demo-artifacts/model.joblib uvicorn app.main:app --port 8000
```

Submit `demo_prediction_request.json` to `POST /predict`. The high sensor values are designed to produce a `failure` prediction, but the precise score is intentionally not asserted because model selection records the actual run metadata.

import joblib
from sensor.platform import train_csv
def test_model_serializes_and_deserializes(dataset, tmp_path):
    _, path = dataset; train_csv(path, tmp_path)
    bundle = joblib.load(tmp_path / "model.joblib")
    assert bundle["metadata"]["features"]

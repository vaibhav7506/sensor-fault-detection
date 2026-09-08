from pathlib import Path

import joblib
import pandas as pd

from sensor.platform import train_csv


def test_checked_in_demo_trains_and_flags_high_risk_request(tmp_path):
    dataset = Path(__file__).parents[1] / "examples" / "demo_sensor_data.csv"
    metadata = train_csv(dataset, tmp_path)
    bundle = joblib.load(tmp_path / "model.joblib")
    request = pd.DataFrame(
        [{"sensor_pressure": 100, "sensor_temperature": 91, "vibration_rms": 8.9}]
    )
    probability = float(
        bundle["model"].predict_proba(bundle["preprocessor"].transform(request))[0, 1]
    )

    assert metadata["features"] == list(request.columns)
    assert probability >= metadata["threshold"]

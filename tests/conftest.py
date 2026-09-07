import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# The project is intentionally runnable directly from a source checkout.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(42); n = 180
    x = rng.normal(size=(n, 4)); y = ((x[:, 0] + .7 * x[:, 1] + rng.normal(0, .6, n)) > 1).astype(int)
    frame = pd.DataFrame(x, columns=["sensor_a", "sensor_b", "sensor_c", "sensor_d"]); frame.loc[0, "sensor_a"] = np.nan; frame["class"] = y
    path = tmp_path / "sensor.csv"; frame.to_csv(path, index=False)
    return frame, path

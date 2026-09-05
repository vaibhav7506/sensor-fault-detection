import numpy as np
from sensor.platform import choose_threshold, metrics
def test_threshold_defaults_without_costs_and_metrics_use_probabilities():
    y = np.array([0, 0, 1, 1]); p = np.array([.1, .2, .8, .9])
    assert choose_threshold(y, p) == .5
    assert metrics(y, p, .5)["pr_auc"] == 1.0

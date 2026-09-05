from sensor.platform import train_csv
def test_training_serializes_model_and_real_metrics(dataset, tmp_path):
    _, path = dataset; metadata = train_csv(path, tmp_path / "model")
    assert (tmp_path / "model" / "model.joblib").exists()
    assert set(["f1", "precision", "recall", "roc_auc", "pr_auc"]) <= set(metadata["test_metrics"])

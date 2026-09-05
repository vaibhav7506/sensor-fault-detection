from sensor.platform import drop_zero_std_columns, validate_frame
def test_validation_and_zero_variance(dataset):
    frame, _ = dataset; frame["constant"] = 1
    report = validate_frame(frame); cleaned, dropped = drop_zero_std_columns(frame)
    assert "constant" in report["zero_variance_columns"] and dropped == ["constant"] and "constant" not in cleaned

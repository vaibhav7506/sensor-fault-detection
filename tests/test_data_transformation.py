from sensor.platform import split_data
def test_validation_and_test_are_not_resampled(dataset):
    frame, _ = dataset; splits = split_data(frame)
    # The training procedure alone invokes SMOTETomek; holdout sizes are immutable.
    assert len(splits.validation) == 36 and len(splits.test) == 36

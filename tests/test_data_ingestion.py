from sensor.platform import split_data
def test_split_is_reproducible_and_stratified(dataset):
    frame, _ = dataset; one, two = split_data(frame), split_data(frame)
    assert one.train.equals(two.train)
    for part in (one.train, one.validation, one.test): assert abs(part["class"].mean() - frame["class"].mean()) < .06

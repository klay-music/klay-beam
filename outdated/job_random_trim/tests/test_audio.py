import numpy as np
from job_random_trim.audio import random_crop


def test_random_crop():
    SR = 10

    def create_x_seconds_of_data(x, sr=SR, ch=2):
        single_channel = np.arange(x * sr)
        return np.tile(single_channel, (ch, 1))

    # Verify my dummy data generator works as intended
    dummy_result = create_x_seconds_of_data(0.5)
    expected_result = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
    assert np.array_equal(dummy_result, expected_result)

    ninety_seconds_of_data = create_x_seconds_of_data(90)
    assert ninety_seconds_of_data.shape == (2, 900)

    ninety_seconds_of_data_cropped = random_crop(ninety_seconds_of_data, sr=SR)
    assert ninety_seconds_of_data_cropped.shape == (2, 900)

    long_data = create_x_seconds_of_data(100)
    long_data_cropped = random_crop(long_data, sr=SR)
    assert long_data_cropped.shape == (2, 900)

    too_short_data = create_x_seconds_of_data(19.8)
    too_short_data_cropped = random_crop(too_short_data, sr=SR)
    assert too_short_data_cropped is None

    twenty_seconds_of_data = create_x_seconds_of_data(20)
    twenty_seconds_of_data_cropped = random_crop(twenty_seconds_of_data, sr=SR)
    assert twenty_seconds_of_data_cropped.shape == (2, 200)

    sixty_seconds_of_data = create_x_seconds_of_data(60)
    sixty_seconds_of_data_cropped = random_crop(sixty_seconds_of_data, sr=SR)
    assert sixty_seconds_of_data_cropped.shape == (2, 600)

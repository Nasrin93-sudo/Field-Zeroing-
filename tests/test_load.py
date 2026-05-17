from Field_zeroing.load import load_data
import pytest
import numpy as np


def test_load_empty_dataset(mocker, caplog):
    """
    Test that a file with no valid datasets logs an error,
    then successfully loads valid data afterward.
    """

    # First file: empty lines
    # Second file: valid data
    inputs = iter([
        "sample_data/test_data1.dat",
        "sample_data/example_input.dat"
    ])

    mocker.patch(
        "builtins.input",
        side_effect = inputs
    )

    result = load_data()

    # Check logged error
    assert "No valid lines of data found in the file" in caplog.text

    # Check valid result eventually returned
    assert isinstance(result, list)


def test_load_valid_data(mocker):
    """
    Test valid dataset loading.
    """

    mocker.patch(
        "builtins.input",
        return_value="sample_data/example_input.dat"
    )

    result = load_data()

    assert isinstance(result, list)
    assert len(result) > 0

    # Optional: check elements are numpy arrays
    assert all(isinstance(arr, np.ndarray) for arr in result)


def test_load_nonexistent_file(mocker, caplog):
    """
    Test nonexistent file logs an error,
    then valid input succeeds.
    """

    inputs = iter([
        "sample_data/nonexistent_file.dat",
        "sample_data/example_input.dat"
    ])

    mocker.patch(
        "builtins.input",
        side_effect = inputs
    )

    result = load_data()

    # Check error was logged
    assert "not found" in caplog.text

    # Check function eventually succeeded
    assert isinstance(result, list)
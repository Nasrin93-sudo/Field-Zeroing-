from Field_zeroing.load import load_data
import pytest
from pathlib import Path
import numpy as np

def test_load_empty_dataset():
    result = load_data(str("sample_data/test_data1.dat"))
    assert result == "Warning: Empty dataset skipped."

def test_load_valid_data():
    result = load_data(str("sample_data/example_input.dat"))
    assert isinstance(result, list) 



def test_load_nonexistent_file():
    result = load_data("sample_data/nonexistent_file.dat")
    assert isinstance(result, str)
    assert "not found" in result
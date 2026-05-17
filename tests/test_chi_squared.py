from Field_zeroing.chi_squared import fitted_param
import numpy as np  

def test_fitted_param():
    class MockMinuit:
        def __init__(self, values, fval):
            self.values = values
            self.fval = fval

    # Mock data for testing
    mock_values = {
        "Bx_0": 1.0,
        "By_0": 2.0,
        "Bz_0": 3.0,
        "c_x": 0.5,
        "c_y": 1.0,
        "c_z": 1.5
    }
    mock_fval = 10.0
    m = MockMinuit(mock_values, mock_fval)
    experimental_B = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])  # Example experimental data

    I_0, B0_fit, c_fit, chi_squared, reduced_chi_squared = fitted_param(m, experimental_B)

    # Assertions to check the correctness of the output
    assert np.allclose(I_0, -np.array([1.0/0.5, 2.0/1.0, 3.0/1.5]))
    assert np.allclose(B0_fit, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(c_fit, np.array([0.5, 1.0, 1.5]))
    assert chi_squared == mock_fval
    assert reduced_chi_squared == mock_fval / (len(experimental_B) - len(mock_values))              
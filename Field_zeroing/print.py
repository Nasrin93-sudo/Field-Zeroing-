#!/usr/bin/env python
# coding: utf-8


from iminuit import Minuit
from iminuit.cost import LeastSquares
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import skew, kurtosis

#Model function
def model(
    set_currents: np.ndarray, 
    Bx_0: float, 
    By_0: float, 
    Bz_0: float, 
    c_x: float, 
    c_y: float, 
    c_z: float
) -> np.ndarray:
    """
    Compute the total magnetic field magnitude for each set of currents.

    Parameters
    ----------
    set_currents : np.ndarray
        Array of shape (3, N) containing currents in the x, y, and z directions.
    Bx_0, By_0, Bz_0 : float
        Background magnetic field components.
    c_x, c_y, c_z : float
        Calibration coefficients relating current to field.

    Returns
    -------
    np.ndarray
        Magnetic field magnitude for each current set, shape (N,).
    """
    B0 = np.array([Bx_0, By_0, Bz_0])[:, None]
    c = np.array([c_x, c_y, c_z])[:, None]
    return np.sqrt(np.sum((c * set_currents + B0)**2, axis=0))


#### Printing results
def printing(
    data: list[np.ndarray], 
    set_currents: np.ndarray,
    I_0: np.ndarray, 
    B0_fit: np.ndarray, 
    c_fit: np.ndarray, 
    chi_squared: float, 
    reduced_chi_squared: float
) -> None:
    """
    Prints all of the required information.
    """
    print("\n\n","=== Magnetic Field Fit Analysis ===", "\n\n")
    print("To get ZF set:","\n", "\t X ZF setting =", f"{I_0[0]:.8e}", "A",          "\n", "\t Y ZF setting =", f"{I_0[1]:.8e}", "A", "\n", "\t Z ZF setting =", I_0[2], "DAC", "\n" )
    
    calculated_B = model(set_currents, B0_fit[0], B0_fit[1], B0_fit[2], c_fit[0], c_fit[1], c_fit[2])
    residuals = (data[:,4] - calculated_B) * 1000

    Bcalc = pd.DataFrame({
        "Run Numbers":data[:,0].astype(int),
        "B(G)": data[:,4],
        "Bcalc(G)" : calculated_B ,
        "Err(mG)" : residuals
    })

    df_str = Bcalc.to_string(index=False)    # Convert DataFrame to string with no index

    lines = df_str.split('\n')    # Split the string into lines

    # Print header, underline, and the rest
    if lines:
        print(lines[0])  # Header row
        print('-' * len(lines[0]))  # Underline
        for line in lines[1:]:
            print(line)

    print("\n")

    residual_statistics = {
        "Minimum": np.min(residuals),
        "Index of minimum": np.argmin(residuals),
        "Maximum": np.max(residuals),
        "Index of maximum": np.argmax(residuals),
        "Sum": np.sum(residuals),
        "Mean": np.mean(residuals),
        "Standard deviation": np.std(residuals, ddof=1),  # Sample std. dev.
        "Skewness": skew(residuals),
        "Kurtosis": kurtosis(residuals, fisher=True),  # Excess kurtosis
        "Variance": np.var(residuals, ddof=1),
        "Average deviation": np.mean(np.abs(residuals - np.mean(residuals))),
    }
    
    stats_items = list(residual_statistics.items())
    half = (len(stats_items) + 1) // 2  # ensures the last item is included

    print("Statistics on Residuals:", "\n")
    for left, right in zip(stats_items[:half], stats_items[half:] + [("", "")] * (half - len(stats_items[half:]))):
        print(f"{left[0]:20} = {left[1]:10.4g}    |    {right[0]:20} = {right[1]:10.4g}" if right[0] else 
              f"{left[0]:20} = {left[1]:10.4g}")
    print("\n")

    print(f"{'χ²':15} = {chi_squared:>12.4f}")
    print(f"{'Reduced χ²':15} = {float(reduced_chi_squared):>12.4f}" if not isinstance(reduced_chi_squared, str)
          else f"{'Reduced χ²':15} = {reduced_chi_squared}")



#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from iminuit import Minuit
from iminuit.cost import LeastSquares
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import skew, kurtosis


# Model function
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


# Fit data
def fitting(
    set_currents: np.ndarray, 
    experimental_B: np.ndarray, 
    B_error: np.ndarray  
) -> Minuit:
    """
    Fit the magnetic field model to experimental data using least squares.

    Parameters
    ----------
    set_currents : np.ndarray
        Currents array of shape (3, N) for x, y, z directions.
    B : np.ndarray
        Measured total magnetic field values (shape: N,).
    B_error : np.ndarray
        Uncertainties on B (shape: N,).

    Returns
    -------
    Minuit
        Fitted Minuit object containing:
        - optimal parameter values
        - uncertainties (via Hesse / Minos)
        - covariance matrix
        - minimization status
        - fit diagnostics (χ², convergence info)
    """
    least_squares = LeastSquares(set_currents, experimental_B, B_error, model)
    least_squares._ndim = 3  # required for vector input
    m = Minuit(
        least_squares,
        Bx_0=0.0, By_0=0.0, Bz_0=0.0,
        c_x=1.0, c_y=1.0, c_z=1.0
    )
    m.migrad()
    m.minos()
    return m


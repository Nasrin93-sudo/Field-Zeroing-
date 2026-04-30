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

# Fitted parameters 
def fitted_param(
    m: Minuit, 
    experimental_B: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Extract fitted parameters and compute derived quantities with uncertainties.

    Parameters
    ----------
    m : Minuit
        Fitted Minuit object after minimization.
    experimental_B : np.ndarray
        Experimental magnetic field values of shape (N,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, float | str]
        I_0 : np.ndarray
            Currents required to obtain zero environmental field.
        B0_fit : np.ndarray
            Fitted environmental field components [Bx_0, By_0, Bz_0].
        c_fit : np.ndarray
            Fitted calibration coefficients [c_x, c_y, c_z].
        reduced_chi_squared : float or a string representing an error
            Reduced chi-squared of the fit.
    """
    p = m.values  
    
    B0_keys = ["Bx_0", "By_0", "Bz_0"]
    c_keys = ["c_x", "c_y", "c_z"]

    B0_fit = np.array([p[k] for k in B0_keys])
    c_fit  = np.array([p[k] for k in c_keys])
    I_0 = - B0_fit / c_fit
    
    # reduced Chi-squared
    chi_squared = m.fval
    if (len(experimental_B) - len(p)) > 0:
        reduced_chi_squared = chi_squared / (len(experimental_B) - len(p))
    else:
        reduced_chi_squared = ("Undefined! The number of datapoints "
        "needs to be larger than the number of the fitted parameters.")
        
    return I_0, B0_fit, c_fit, chi_squared, reduced_chi_squared   


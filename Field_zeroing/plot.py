#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import skew, kurtosis
import logging

logging.basicConfig(level=logging.INFO)


#### PLOTTING results

# Processes data for the purpose of plotting
def process_plot(data: np.ndarray,
                 set_I_x: np.ndarray,
                 set_I_y: np.ndarray,
                 set_I_z: np.ndarray,
                 B0_fit: np.ndarray,
                 c_fit: np.ndarray,
                 ) -> plt.Figure:
    """
    Extract three scan-specific subsets from an input data array.

    The function receives a data array and extracts the following three arrays from it:

    1. Rows where columns 3 and 4 are constant while column 2 changes.
    2. Rows where columns 2 and 4 are constant while column 3 changes.
    3. Rows where columns 2 and 3 are constant while column 4 changes.
    
    Then plots total external magnetic field vs current for each x, y, and z direction along with fitted lines.
    Returns
    -------
    Returns 1, 2, or 3 figures.
    """
    df = pd.DataFrame(data, columns = ['run_num', 'x', 'y', 'z', 'B', 'err'])
    x_scan = max(df.groupby(['y', 'z']), key=lambda g: len(g[1]))[1].to_numpy()
    y_scan = max(df.groupby(['x', 'z']), key=lambda g: len(g[1]))[1].to_numpy()
    z_scan = max(df.groupby(['x', 'y']), key=lambda g: len(g[1]))[1].to_numpy()
    
    num_non_empty = sum(arr.size > 0 for arr in [x_scan, y_scan, z_scan]) # figue preparation
    
    fig, axes = plt.subplots(1, num_non_empty, figsize=(5 * num_non_empty, 4))
    if num_non_empty == 1:
        axes = [axes]     
    
    # Define scan data and labels for looping
    scans = {
        'x': (x_scan, 1, set_I_x, 'Current (A)'),
        'y': (y_scan, 2, set_I_y, 'Current (A)'),
        'z': (z_scan, 3, set_I_z, 'Current (DAC)')
    }
   
    # Precompute fit lines
    fit_currents = {
        'x': np.linspace(set_I_x.min(), set_I_x.max(), 500),
        'y': np.linspace(set_I_y.min(), set_I_y.max(), 500),
        'z': np.linspace(set_I_z.min(), set_I_z.max(), 500)
    }  
    
    fit_curves = {
        'x': lambda I: np.sqrt((c_fit[0] * I + B0_fit[0]) ** 2 + 
                               (c_fit[1] * x_scan[0,2] + B0_fit[1]) ** 2 + 
                               (c_fit[2] * x_scan[0,3] + B0_fit[2]) ** 2),

        'y': lambda I: np.sqrt((c_fit[0] * y_scan[0,1] + B0_fit[0]) ** 2 + 
                               (c_fit[1] * I + B0_fit[1]) ** 2 + 
                               (c_fit[2] * y_scan[0,3] + B0_fit[2]) ** 2),

        'z': lambda I: np.sqrt((c_fit[0] * z_scan[0,1] + B0_fit[0]) ** 2 + 
                               (c_fit[1] * z_scan[0,2] + B0_fit[1]) ** 2 + 
                               (c_fit[2] * I + B0_fit[2]) ** 2)
    }
    
    # Plot in loop
    plot_idx = 0
    for idx, (axis, (scan_data, col, I_data, xlabel)) in enumerate(scans.items()):
        if scan_data.size == 0:
            continue    
        ax = axes[plot_idx]
        plot_idx += 1
        I_fit = fit_currents[axis]
        B_fit = fit_curves[axis](I_fit)

        ax.errorbar(scan_data[:, col], scan_data[:, 4], yerr=scan_data[:, 5],
                    fmt='o', capsize=3, label=f'{axis}-Data', color='blue')
        ax.plot(I_fit, B_fit, color='red',
                label=rf'Fit')

        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel('Magnetic Field (G)', fontsize=14)
        ax.set_title(f'{axis.upper()} - Scan', fontsize=16)
        ax.grid(True)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()

    if num_non_empty < 3:
        logging.warning(
            f"Warning: Only {num_non_empty} dataset(s) found. Output will reflect that."
        )
    
    return fig, axes


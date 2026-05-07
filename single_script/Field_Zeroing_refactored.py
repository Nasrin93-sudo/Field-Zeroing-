#!/usr/bin/env python
# coding: utf-8

# In[18]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


from iminuit import Minuit
from iminuit.cost import LeastSquares
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import skew, kurtosis
import logging

logging.basicConfig(level=logging.INFO)

def load_data() -> list[np.ndarray]:
    """
    Load data from a user-provided file path.

    The function reads a text-based data file, removes empty lines and comment lines
    (lines starting with '#'), and parses each line into a NumPy array.

    Each valid line is converted into a NumPy array and stored in a list.

    Returns
    ---------------
    list of np.ndarray
        A list containing parsed lines as NumPy arrays.

    Notes
    ---------------
    - Invalid or empty lines are skipped with a warning.
    - Non-text or unsupported file formats (e.g. PNG, PDF) will cause reading or parsing errors.
    - The function repeatedly prompts the user until a valid file is successfully processed.
    """
    
    while True:
        file_path = input("Please enter the full path to your data file: ").strip()
            
        if not os.path.exists(file_path):
            logging.error(f"Error: File '{file_path}' not found. Please try again.")
            continue
            
        try:
            with open(file_path, 'r') as file:
                
                # Read all lines, strip whitespace, and remove empty lines and comment lines
                lines = [line.strip() for line in file if line.strip() and not line.strip().startswith('#')]  
                    
                
                # Parse each dataset
                parsed_lines = []
                for line in lines:
                    try:
                        data = np.genfromtxt([line], dtype=float)
                        if data.size == 0:
                            logging.warning("Warning: Empty dataset skipped.")
                            continue
                        parsed_lines.append(data)
                    except Exception as e:
                        logging.error(f"Error parsing dataset: {e}")
                        continue
                
                if not parsed_lines:
                    logging.error("Error: No valid lines of data found in the file.")
                    continue
                return parsed_lines
        except Exception as e:
            logging.error(f"Error reading file: {e}. Please check the format and try again.")

                        


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
        logging.warning("Undefined! The number of datapoints "
                        "needs to be larger than the number of the fitted parameters.")
        reduced_chi_squared = "Undefined!"

    return I_0, B0_fit, c_fit, chi_squared, reduced_chi_squared   



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




#### PLOTTING results

# Processes data for the purpose of plotting
def process_plot(data: np.ndarray,
                 set_I_x: np.ndarray,
                 set_I_y: np.ndarray,
                 set_I_z: np.ndarray,
                 B0_fit: np.ndarray,
                 c_fit: np.ndarray,
                 ) -> None:
    """
    Extract three scan-specific subsets from an input data array.

    The function receives a data array and extracts the following three arrays from it:

    1. Rows where columns 3 and 4 are constant while column 2 changes.
    2. Rows where columns 2 and 4 are constant while column 3 changes.
    3. Rows where columns 2 and 3 are constant while column 4 changes.
    
    Then plots total external magnetic field vs current for each x, y, and z direction along with fitted lines.
    Returns
    -------
    Does not retyrn anything.
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
    plt.draw()
    plt.pause(0.001)
    
    if num_non_empty < 3:
        logging.warning(f"Warning: Only {num_non_empty} dataset(s) found. Output will reflect that.")

    input("Press Enter to close the plot and exit...")


def main():
    """
    Run full analysis pipeline
    """
    # step 1: Load
    data_set = np.array(load_data())
    run_number, set_I_x, set_I_y, set_I_z, experimental_B, B_error = data_set.T
    set_currents = np.array([set_I_x, set_I_y, set_I_z])
    
    # step 2: Fit
    fit_data = fitting(set_currents, experimental_B, B_error)
    m = fitting(set_currents, experimental_B, B_error)
    
    # step 3:  Chi_squared
    fitted_parameters = fitted_param(m, experimental_B)
    I_0, B0_fit, c_fit, chi_squared, reduced_chi_squared = fitted_param(m, experimental_B)   
    
    # step 4: 
    printing (data_set, set_currents, I_0, B0_fit, c_fit, chi_squared, reduced_chi_squared)
    
    # step 5: plotting
    process_plot(data_set, set_I_x, set_I_y, set_I_z, B0_fit, c_fit)

if __name__ == "__main__":
    main()

# In[ ]:


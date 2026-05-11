#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import matplotlib.pyplot as plt
from Field_zeroing.load import load_data
from Field_zeroing.chi_squared import fitted_param
from Field_zeroing.fit import fitting
from Field_zeroing.fit import model
from Field_zeroing.plot import process_plot
from Field_zeroing.print import printing

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
    fig, axes =process_plot(
        data_set, 
        set_I_x, 
        set_I_y, 
        set_I_z, 
        B0_fit, 
        c_fit
    )

    fig.savefig("figures/output_magnetic_field_scan.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

# Function to load data from file with user input
def load_data():
    while True:
        file_path = input("Please enter the full path to your data file: ").strip()
            
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found. Please try again.")
            continue
        try:
            with open(file_path, 'r') as file:
                # Read all lines, strip whitespace, and remove empty lines and comment lines
                lines = [line.strip() for line in file if line.strip() and not line.strip().startswith('#')]  
                    
                datasets = []
                for line in lines:
                    datasets.append(line)
                    
                
                # Parse each dataset
                parsed_datasets = []
                for run in datasets:
                    try:
                        data = np.genfromtxt(run.splitlines(), dtype=float)
                        if data.size == 0:
                            print("Warning: Empty dataset skipped.")
                            continue
                        parsed_datasets.append(data)
                    except Exception as e:
                        print(f"Error parsing dataset: {e}")
                        continue
                
                if not parsed_datasets:
                    print("Error: No valid datasets found in the file.")
                    continue
                return parsed_datasets
        except Exception as e:
            print(f"Error reading file: {e}. Please check the format and try again.")
            
print("=== Magnetic Field Fit Analysis ===", "\n\n")
data_set = load_data()


def process_data(data):
    df = pd.DataFrame(data)

    # Step 1: Keep only rows with values in col3 that appear more than 2 times
    freq_col3 = df[3].value_counts()
    values_over_2 = freq_col3[freq_col3 > 2].index.tolist()
    df_filtered = df[df[3].isin(values_over_2)].copy()

    # Step 2: Within df_filtered, find most frequent value in column 2
    freq_col2 = df_filtered[2].value_counts()
    value_over_1 = freq_col2[freq_col2 > 1].index.tolist()
    
    
    # Step 3: Bring those rows to top
    df_top = df_filtered[df_filtered[2].isin(value_over_1)].copy()
    df_rest_filtered = df_filtered[~df_filtered[2].isin(value_over_1)].copy()
    
    
         ## handling the case when we have an x_scan run and a y_scan run with the same y currents and different z currents
   # if (df_top[3].value_counts() == 1).any():
    #    freq_z_top = df_top[3].value_counts()
     #   rare = freq_z_top[freq_z_top == 1].index.tolist()
      #  rare_row = df_top[df_top[3].isin(rare)]
       # df_top = df_top[~df_top[3].isin(rare)]
        #df_rest_filtered = pd.concat([df_rest_filtered, rare_row], ignore_index=True)
        
         ## removing single-occurrences
    #y_scans_freq = df_rest_filtered[1].value_counts()
    #single_occurrences = y_scans_freq[y_scans_freq == 1].index.tolist()
    #df_rest_filtered = df_rest_filtered[~df_rest_filtered[1].isin(single_occurrences)]

    
    # Step 4: Remaining rows from original df that were not in df_filtered
    df_remaining = df[~df.index.isin(df_filtered.index)]
        ## removing single-occurrences
    #z_scans_freq = df_remaining[1].value_counts()
    #single_occurrences = z_scans_freq[z_scans_freq == 1].index.tolist()
    #df_remaining = df_remaining[~df_remaining[1].isin(single_occurrences)]

    # Step 5: Concatenate all parts
    df_final = pd.concat([df_top, df_rest_filtered, df_remaining], ignore_index=True)

    # Step 6: Convert to list of arrays
    df_top = df_top.values
    df_rest_filtered = df_rest_filtered.values
    df_remaining = df_remaining.values
    df_final = df_final.values
    
    return [df_final, df_top, df_rest_filtered, df_remaining]

    
data_set, x_scan, y_scan, z_scan = process_data(data_set)

num_non_empty = sum(arr.size > 0 for arr in [x_scan, y_scan, z_scan])

# Initialize figure for subplots
fig, axes = plt.subplots(1, num_non_empty, figsize=(5 * num_non_empty, 4))
if num_non_empty == 1:
    axes = [axes] 

run_number = data_set[:, 0]
I_x = data_set[:, 1]
I_y = data_set[:, 2]
I_z = data_set[:, 3]
B = data_set[:, 4]
B_error = data_set[:, 5]

II = [I_x, I_y, I_z]

# Model function
def model(II, Bx_0, By_0, Bz_0, c_x, c_y, c_z):
    I_x, I_y, I_z = II[0], II[1], II[2]
    return np.sqrt((c_x * I_x + Bx_0) ** 2 + 
                   (c_y * I_y + By_0) ** 2 + 
                   (c_z * I_z + Bz_0) ** 2)



# Fit
least_squares = LeastSquares(II, B, B_error, model)
least_squares._ndim = 3
m = Minuit(least_squares,
           Bx_0=0.0, By_0=0.0, Bz_0=0.0,
           c_x=1.0, c_y=1.0, c_z=1.0)

m.migrad()
m.minos()

# Fitted parameters
Bx_0_fit, By_0_fit, Bz_0_fit  = m.values["Bx_0"], m.values["By_0"], m.values["Bz_0"]
c_x_fit, c_y_fit, c_z_fit = m.values["c_x"], m.values["c_y"], m.values["c_z"]

Bx_0_err_lower, Bx_0_err_upper = m.merrors["Bx_0"].lower, m.merrors["Bx_0"].upper
By_0_err_lower, By_0_err_upper = m.merrors["By_0"].lower, m.merrors["By_0"].upper
Bz_0_err_lower, Bz_0_err_upper = m.merrors["Bz_0"].lower, m.merrors["Bz_0"].upper
c_x_err_lower, c_x_err_upper = m.merrors["c_x"].lower, m.merrors["c_x"].upper
c_y_err_lower, c_y_err_upper = m.merrors["c_y"].lower, m.merrors["c_y"].upper
c_z_err_lower, c_z_err_upper = m.merrors["c_z"].lower, m.merrors["c_z"].upper


# Calculate I_0 and its asymmetric error
scan_flags = {
    'x': x_scan,
    'y': y_scan,
    'z': z_scan
}

Ix_00 = -Bx_0_fit / c_x_fit
Iy_00 = -By_0_fit / c_y_fit
Iz_00 = -Bz_0_fit / c_z_fit

Ix_0 = "No data!"
Iy_0 = "No data!"
Iz_0 = "No data!"

for axis in 'xyz':
    if scan_flags[axis].any():
        B_fit = locals()[f'B{axis}_0_fit']
        B_err_lower = locals()[f'B{axis}_0_err_lower']
        B_err_upper = locals()[f'B{axis}_0_err_upper']
        c_fit = locals()[f'c_{axis}_fit']
        c_err_lower = locals()[f'c_{axis}_err_lower']
        c_err_upper = locals()[f'c_{axis}_err_upper']

        locals()[f'I{axis}_0'] = -B_fit / c_fit
        locals()[f'I{axis}_0_err_lower'] = np.sqrt((B_err_lower / c_fit) ** 2 + ((B_fit / c_fit ** 2) * c_err_lower) ** 2)
        locals()[f'I{axis}_0_err_upper'] = np.sqrt((B_err_upper / c_fit) ** 2 + ((B_fit / c_fit ** 2) * c_err_upper) ** 2)



# Chi-squared
chi_squared = m.fval
if (len(B) - len(m.values)) > 0:
    reduced_chi_squared = chi_squared / (len(B) - len(m.values))
else:
    reduced_chi_squared = "Undefined! The number of datapoints needs to be larger than the number of the fitted parameters."

#### Printing results

def smart_format(val):
    if isinstance(val, (int, float)):
        return f"{val:.8e}"  # scientific notation
    else:
        return str(val)   
        
print("\n\n To get ZF set:","\n", "\t X ZF setting =", smart_format(Ix_0), "A", "\n", "\t Y ZF setting =", smart_format(Iy_0), "A", "\n",       "\t Z ZF setting =", Iz_0, "DAC", "\n" )

calculated_B = model(II, Bx_0_fit, By_0_fit, Bz_0_fit, c_x_fit, c_y_fit, c_z_fit)
residuals = (data_set[:,4] - calculated_B ) * 1000

Bcalc = pd.DataFrame({
    "Run Numbers":data_set[:,0].astype(int),
    "B(G)": data_set[:,4],
    "Bcalc(G)" : calculated_B ,
    "Err(mG)" : residuals
})

# Convert DataFrame to string with no index
df_str = Bcalc.to_string(index=False)

# Split the string into lines
lines = df_str.split('\n')

# Print header, underline, and the rest
if lines:
    print(lines[0])  # Header row
    print('-' * len(lines[0]))  # Underline
    for line in lines[1:]:
        print(line)

print("\n")

residual_stats = {
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

stats_items = list(residual_stats.items())
half = (len(stats_items) + 1) // 2  # ensures the last item is included

print("Statistics on Residuals:", "\n")
for left, right in zip(stats_items[:half], stats_items[half:] + [("", "")] * (half - len(stats_items[half:]))):
    print(f"{left[0]:20} = {left[1]:10.4g}    |    {right[0]:20} = {right[1]:10.4g}" if right[0] else 
          f"{left[0]:20} = {left[1]:10.4g}")
print("\n")

print(f"{'χ²':15} = {chi_squared:>12.4f}")
print(f"{'Reduced χ²':15} = {float(reduced_chi_squared):>12.4f}" if not isinstance(reduced_chi_squared, str)
      else f"{'Reduced χ²':15} = {reduced_chi_squared}")

#### PLOTTING
# Define scan data and labels for looping
scans = {
    'x': (x_scan, 1, I_x, 'Current (A)'),
    'y': (y_scan, 2, I_y, 'Current (A)'),
    'z': (z_scan, 3, I_z, 'Current (DAC)')
}

# Precompute fit lines
fit_currents = {
    'x': np.linspace(I_x.min(), I_x.max(), 500),
    'y': np.linspace(I_y.min(), I_y.max(), 500),
    'z': np.linspace(I_z.min(), I_z.max(), 500)
}

fit_curves = {
    'x': lambda I: np.sqrt((c_x_fit * I + Bx_0_fit) ** 2 + 
                           (c_y_fit * x_scan[0,2] + By_0_fit) ** 2 + 
                           (c_z_fit * x_scan[0,3] + Bz_0_fit) ** 2),
    'y': lambda I: np.sqrt((c_x_fit * y_scan[0,1] + Bx_0_fit) ** 2 + 
                           (c_y_fit * I + By_0_fit) ** 2 + 
                           (c_z_fit * y_scan[0,3] + Bz_0_fit) ** 2),
    'z': lambda I: np.sqrt((c_x_fit * z_scan[0,1] + Bx_0_fit) ** 2 + 
                           (c_y_fit * z_scan[0,2] + By_0_fit) ** 2 + 
                           (c_z_fit * I + Bz_0_fit) ** 2)
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
    print(f"Warning: Only {num_non_empty} dataset(s) found. Output will reflect that.")

input("Press Enter to close the plot and exit...")

# In[ ]:


# In[1]:


import os
os.getcwd()


# In[ ]:





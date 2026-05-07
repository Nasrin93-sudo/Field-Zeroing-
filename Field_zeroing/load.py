#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from iminuit import Minuit
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
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

                        


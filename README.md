# Field Zeroing Analysis

This repository contains two implementations of a magnetic field analysis and fitting pipelines:

- A **single-file script** for simple portable use
- A **modular version** for developement, maintainability, and extension

-------------------------------------------------------------------------------

### Repository Structure
<pre>
Field Zeroing/
|__ main.py            #Entry point for modular pipeline
|__ Field_zeroing      #Modular package (core functions)
|  |__ load.py
|  |__ fit.py
|  |__ plot.py
|  |__ init.py
|  |__ print.py
|  |__ chi_squared.py
|
|__ single_script/
|  |__ Field_Zeroing_refracted.py   #standalone version (all-in-one script)
|
|__ sample_data/
|  |__ example_input.dat            #Example dataset for testing
</pre>


-------------------------------------------------------------------------------

### How to Run
1. Single-file version (quick use)
This version is self-contained and easy to run on any system:

**bash:**

python single_script/Field_Zeroing_refactored.py

2. Modular version (recommended for developement)
run the full pipeline:

**bash:**

python main.py

Or in Jupyter:

%run main.py

-------------------------------------------------------------------------------
### Sample Input Data
A sample dataset us provided for testing:
sample_data/example_input.dat

Format (only column order matters):

Column 1: Run number (labels every round of magnetic field measurement)
Column 2: set current for x direction
Column 3: set current for y direction
Column 4: set current for z direction
Column 5: Measured magnetic field
Column 6: Uncertainty of the measured magnetic field

This file can be used to verify that the pipeline runs correctly.

-------------------------------------------------------------------------------

### Requirements:
Install dependencies with:

**bash:**

pip install numpy scipy pandas matplotlib iminuit os

-------------------------------------------------------------------------------

### Notes:
Both versions implement the same analysis workflow.

-------------------------------------------------------------------------------

### Author notes:
This project was developed for magnetic field analysis and model fitting using Python-based scientific computing tools.

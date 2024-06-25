# Postprocessing_of_protcomplex_sims_Dissertation
This is my postprocessing code for protein complex simulations. Not everything is included and fresh.

# CytoCast PostProcessing Tool

## Overview

This project is a graphical user interface (GUI) application designed for the post-processing of CytoCast experiment data. It provides various functionalities such as loading experiment data, performing Principal Component Analysis (PCA), K-means clustering, generating heatmaps, and more. The tool is built using Python's `tkinter` library for the GUI and `matplotlib` for plotting.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Input Fields](#input-fields)
4. [Contributing](#contributing)
5. [License](#license)

## Installation

To install and set up this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/misma2/Postprocessing_of_protcomplex_sims_Dissertation.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Postprocessing_of_protcomplex_sims_Dissertation
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the `main.py` file:
```bash
python main.py
```

This will open the GUI window where you can interact with the various functionalities provided by the tool.

## Input Fields

### Main Buttons and Functionalities

- **Load Data**: Load experiment data files.
- **HeatMap**: Generate a heatmap using input and output data files.
- **Distances**: Calculate and plot distances between wild-type and mutant data.
- **PCA**: Perform Principal Component Analysis on input and output data files.
- **K-means**: Perform K-means clustering on the loaded data with specified `k` and random state values.
- **Plot Pies**: Generate pie charts or bar charts for specified regions using wild-type and mutant data files.
- **T test**: Perform multi-pairwise T-tests on the data.
- **Feature**: Calculate feature importance.
- **Split Matrix**: Split matrix by type.

### Additional Controls

- **k**: Entry field to specify the number of clusters for K-means clustering.
- **random state**: Entry field to specify the random state for K-means clustering.
- **Bar Chart**: Checkbox to toggle between pie chart and bar chart.
- **All**: Checkbox to include all regions.
- **region1** and **region2**: Entry fields to specify regions for plotting.

### Theme Switch

- **Theme Toggle Button**: Switch between light and dark themes.

### Example Plotting Function

- **Plot Sizes**: Plot complex sizes for a specified index.
- **Complex**: Display the structure of a complex for a specified index.
- **Calc sizes**: Calculate and save complex sizes for wild-type and mutant data.


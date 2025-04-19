# Visualization (`/figures`)

This directory contains scripts for generating and storing visualizations.


## Scripts

### 1. class_distribution.py

This script analyzes and visualizes the distribution of classes in the dataset.

**Functionality:**
- Loads meta data
- Calculates and displays class distribution statistics in the console
- Creates a bar chart visualization of the class distribution

**Output:**
- Console output showing class counts, total samples, and number of classes
- Bar chart image file saved to 'figures/Results/class_distribution.png'

### 2. calibrator_comparison_heatmap.py

This script creates a heatmap to compare the performance of different calibrators against each other.

**Functionality:**
- Loads meta data from
- Calculates pairwise outperformance ratios between calibrators
- Creates a heatmap visualization showing these comparative ratios

**Output:**
- Heatmap image file saved to 'figures/Results/calibrator_comparison_heatmap.png'
- Each cell in the heatmap shows the ratio of cases where the row calibrator outperforms the column calibrator


## Directory Structure

- `figures/` - Main directory for visualization scripts
  - `Results/` - Directory where generated visualization files are saved

## Usage

To generate the class distribution visualization, run from root dir:

```bash
python -m figures.class_distribution
```

To generate the calibrator comparison heatmap, run from root dir:

```bash
python -m figures.calibrator_comparison_heatmap
```



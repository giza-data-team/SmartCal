import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager


config = ConfigurationManager()

output_folder = 'figures/Results'
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Load the dataset
df = pd.read_csv(config.meta_data_file)
class_counts = df['Best_Cal'].value_counts()
calibrators = class_counts.index.tolist()

# Create a matrix to store pairwise ratios
matrix = pd.DataFrame(index=calibrators, columns=calibrators, dtype=float)

for i in calibrators:
    for j in calibrators:
        if i == j:
            matrix.loc[i, j] = np.nan  # Diagonal as NaN
        else:
            count_i = class_counts[i]
            count_j = class_counts[j]
            total = count_i + count_j
            ratio = count_i / total if total != 0 else 0
            matrix.loc[i, j] = ratio

# Plot heatmap
plt.figure(figsize=(20, 12))
ax = sns.heatmap(matrix, annot=True, fmt=".2f",  cmap="YlGnBu",
                 cbar_kws={'label': 'Outperformance Ratio'}, linewidths=0.5, linecolor='white',annot_kws={"size":20})
plt.title("Calibrators Performing Comparison",fontsize=25, pad=20)
plt.xlabel("Main Calibrator",fontsize=20)
plt.ylabel("Calibrator Outperforming",fontsize=20)
plt.xticks(rotation=45, ha='right',fontsize=20)
plt.yticks(rotation=0,fontsize=20)

# Modify the colorbar label and tick font sizes
cbar = ax.collections[0].colorbar
cbar.set_label('Outperformance Ratio', fontsize=20)
cbar.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.savefig('figures/Results/calibrator_comparison_heatmap.png')
plt.show()

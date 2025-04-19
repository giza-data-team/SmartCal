import os
import pandas as pd
import matplotlib.pyplot as plt

from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager


config = ConfigurationManager()

# Load the dataset from CSV
df = pd.read_csv(config.meta_data_file)

output_folder = 'figures/Results'
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Calculate class distribution
class_counts = df['Best_Cal'].value_counts()

# Print class distribution statistics
print("\nClass distribution:")
print(class_counts)
print(f"\nTotal samples: {len(df)}")
print(f"Number of classes: {len(class_counts)}")

# Plot settings
plt.figure(figsize=(12, 8))

# Create bar plot
ax = class_counts.plot(kind='bar', color=['#4C72B0', '#55A868', '#CCB974', '#8172B2'])

# Customize plot
plt.title('Class Distribution - Best Calibrator', fontsize=25, pad=20)
plt.xlabel('Class', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.xticks(rotation=45, ha='right', fontsize=20)
plt.yticks(fontsize=20)

# Add count labels on top of bars
for p in ax.patches:
    ax.annotate(f"{p.get_height()}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 5),
                textcoords='offset points',
                fontsize=20)

# Add grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout and display
plt.tight_layout()
plt.savefig('figures/Results/class_distribution.png')
plt.show()


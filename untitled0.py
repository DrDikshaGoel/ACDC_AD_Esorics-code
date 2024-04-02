import matplotlib.pyplot as plt
import numpy as np

# Data for r1000 setup only
policies = ['G24 (Proposed)', 'G23', 'G50']
categories = ['Independent', 'Positive', 'Negative']
success_chances = [
    [53.69, 52.63, 52.31],
    [47.27, 50.94, 57.97],
    [51.83, 53.76, 54.55]
]

# Values for G50 for comparison
g50_values = [51.83, 53.76, 54.55]

# Calculate standard deviation for error bars
std_dev = np.std(success_chances, axis=0)

# Plot
bar_width = 0.2
index = np.arange(len(categories))

plt.figure(figsize=(10, 6))

for i, policy in enumerate(policies):
    plt.bar(index + (i - 1) * bar_width, success_chances[i], bar_width, yerr=std_dev, capsize=5, label=policy, alpha=0.7)

# Highlight G50 values
plt.scatter(index, g50_values, color='red', label='G50', zorder=5)

# Annotate bars with values
for i in range(len(categories)):
    for j in range(len(policies)):
        plt.text(index[i] + (j - 1) * bar_width, success_chances[j][i] + 1, f'{success_chances[j][i]:.2f}', ha='center', va='bottom')

plt.xlabel('Success Categories')
plt.ylabel('Chances of Success')
plt.title('Chances of Success for r1000 Setup')
plt.xticks(index, categories)
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

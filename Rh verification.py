import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Read data
df = pd.read_csv('Rh.csv')  # Replace path_to_your_file.csv with your file path

# Extract validation Rh and simulated Rh data
x = df['Validation Rh']
y = df['Simulated Rh']

# Calculate linear regression
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Calculate R2 and RMSE
r_squared = r_value ** 2
rmse = np.sqrt(mean_squared_error(x, y))
p = p_value + 0.001
# Number of data points
n = len(df)

# Plot scatter density with gradient color
plt.figure(figsize=(10, 6))

# Use a different color map
hb = plt.hexbin(x, y, gridsize=50, cmap='summer', mincnt=1)
cb = plt.colorbar(hb)
cb.set_label('Counts', fontsize=12)
cb.ax.tick_params(labelsize=12)

# Plot the fit line
x_vals = np.array(plt.gca().get_xlim())
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, '--', color='black', label='Linear fit')

# Add R2 and RMSE to the top left corner
plt.text(0.05, 0.88, f'$R^2$ = {r_squared:.2f}\nP < {p:.3f}\nRMSE = {rmse:.2f}\nN = {n}',
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')

# Set tick marks for the left and bottom axes
ylabel = r'Simulated R$_h$ value (gC m$^{-2}$ yr$^{-1}$)'
xlabel = r'Global R$_h$ value (gC m$^{-2}$ yr$^{-1}$)'
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.xlabel(xlabel, fontsize=14)
plt.ylabel(ylabel, fontsize=14)

# Set tick font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Display all axes but hide tick marks on the right and top axes
ax = plt.gca()
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.tick_params(axis='x', top=False, labeltop=False, direction='in')  # No tick marks on the x-axis
ax.tick_params(axis='y', right=False, labelright=False, direction='in')  # No tick marks on the y-axis

# Move legend to the lower right
plt.legend(loc='lower right', frameon=False, fontsize=15)
plt.savefig(r'path\Rh.jpg', format='jpg', dpi=600, bbox_inches='tight')
plt.show()

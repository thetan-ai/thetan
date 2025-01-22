import numpy as np
import matplotlib.pyplot as plt

# Define the range of n
n = np.linspace(0, 5000, 1000)

# Define the complexity functions
o_n = n
o_n2 = n**2

# Create the figure and axis
fig, ax = plt.subplots(figsize=(16, 8))

# Plot the complexity functions
ax.plot(n, o_n, label='Best $O(n)$', color='blue', linewidth=3)
ax.plot(n, 300 * o_n, label='Avg $O(nk + k^2)$', color='orange', linewidth=3)
ax.plot(n, o_n2, label='Worst $O(n^2)$', color='red', linewidth=3)

# Fill the area between O(n) and O(n^2)
ax.fill_between(n, o_n, o_n2, color='lightgray', alpha=0.5, label='Shaded Region')

# Add labels, legend, and grid
ax.set_xlabel('$n$', fontsize=22)
ax.set_ylabel('Runtime Complexity', fontsize=22)
ax.legend(fontsize=22)
ax.grid(alpha=0.3)

# Adjust ticks
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)

ty = ax.yaxis.get_offset_text()
ty.set_fontsize(18)

# Save the plot
fig.savefig('bigoplot.png')

# Optionally, show the plot (if needed)
# plt.show()





1/0

# Define the range of n
n = np.linspace(0, 5000, 1000)

# Define the complexity functions
o_n = n
o_n2 = n**2

# Create the plot
plt.figure(figsize=(16, 8))
plt.plot(n, o_n, label='Best $O(n)$', color='blue', linewidth=3)
plt.plot(n, 300*o_n, label='Avg $O(nk + k^2)$', color='orange', linewidth=3)
plt.plot(n, o_n2, label='Worst $O(n^2)$', color='red', linewidth=3)

# Fill the area between O(n) and O(n^2)
plt.fill_between(n, o_n, o_n2, color='lightgray', alpha=0.5, label='Shaded Region')

# Add labels and legend
# plt.title('Thetan Berserker  $O(n)$ to $O(n^2)$', fontsize=14)
plt.xlabel('$n$', fontsize=22)
plt.ylabel('Runtime Complexity', fontsize=22)
plt.legend(fontsize=22)
plt.grid(alpha=0.3)
plt.xticks(fontsize=22)
yt = plt.yticks(fontsize=22)


# yaxis.get_offset_text()
# tx.set_fontsize(7)
# # Show the plot
# plt.show()
plt.savefig('bigoplot.png')

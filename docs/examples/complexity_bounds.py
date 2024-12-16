import numpy as np
import matplotlib.pyplot as plt

# Define the range of n
n = np.linspace(0, 5000, 1000)

# Define the complexity functions
o_n = n
o_n2 = n**2

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(n, o_n, label='Best $O(n)$', color='blue', linewidth=2)
plt.plot(n, 300*o_n, label='Avg $O(nk + k^2)$', color='orange', linewidth=2)
plt.plot(n, o_n2, label='Worse $O(n^2)$', color='red', linewidth=2)

# Fill the area between O(n) and O(n^2)
plt.fill_between(n, o_n, o_n2, color='lightgray', alpha=0.5, label='Shaded Region')

# Add labels and legend
# plt.title('Thetan Berserker  $O(n)$ to $O(n^2)$', fontsize=14)
plt.xlabel('$n$', fontsize=12)
plt.ylabel('Runtime Complexity', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# Show the plot
plt.show()

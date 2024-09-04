import numpy as np
from thetan.theta import centroids_from_labels
from thetan.chaining import chainer, chainer2
from thetan.berserker import berserker2
from thetan.viz import plot_centroids_clusters


spiral_data = np.genfromtxt("/home/elef/tmp/SpiralData_Clean_Spread.csv", delimiter=',')
# spiral_data = np.genfromtxt("C:/Users/Eleftherios/tmp/SpiralData_Clean_Spread.csv", delimiter=',')

# first entry is nan
spiral_data = np.delete(spiral_data, 0, 0)
X = spiral_data.copy()

theta_value = 1.0
epsilon_value = 2.0

initial_centroids, initial_labels = berserker2(X, theta_value, all_out=False)

plot_centroids_clusters(X, initial_centroids, initial_labels, title='Berserker2', show=True)

new_labels = chainer(X, initial_centroids, initial_labels, epsilon_value)
new_labels = np.ascontiguousarray(new_labels, dtype='i8')
# new_labels are not guaranteed to be 0, 1, 2, 3, ... you can have 0, 5, 8

# indices_arg = np.argsort(new_labels)
# sorted_labels = new_labels[indices_arg]

new_centroids = centroids_from_labels(X, new_labels)


plot_centroids_clusters(X, centroids=new_centroids, labels=new_labels, title='Chainer', show=True)

t = 1



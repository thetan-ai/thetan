from numpy import genfromtxt
import numpy as np
from robotikos.fcn import fcn


class DataSequence:

    def __init__(self, data):

        self.data = data

    def __getitem__(self, i):
        return self.data[i][None, :]

    def __len__(self):
        return self.data.shape[0]


# change data directory to yours
# TO DO automatically find the path to data


# Eaxmple 1
# 3 Spirals data
my_data = genfromtxt("/Users/bramshqamarchandio/robotikos/data/SpiralData_Clean_Spread.csv", delimiter=',')
# first entry is nan
my_data = np.delete(my_data, 0, 0)
x_data = my_data.copy()

''' last dimension is different for each cluster presumably it's the label
already provided in the data making it 0 for all '''

for i in range(len(x_data)):
    x_data[i][2] = 0.00

dataseq = DataSequence(x_data)

fcn(data=dataseq, qbx_thresholds=[0.75], chaining_thr=1.5, viz=True)


# Eaxmple 2
# compound data
my_data = np.loadtxt("/Users/bramshqamarchandio/robotikos/data/Compound.txt")
x_data = my_data.copy()

''' last dimension is different for each cluster presumably it's the label
already provided in the data making it 0 for all '''

for i in range(len(x_data)):
    x_data[i][2] = 0.00

dataseq = DataSequence(x_data)

fcn(data=dataseq, qbx_thresholds=[1], chaining_thr=1.7, viz=True)

# Eaxmple 3
# Aggregation data
my_data = np.loadtxt("/Users/bramshqamarchandio/robotikos/data/Aggregation.txt")
x_data = my_data.copy()

''' last dimension is different for each cluster presumably it's the label
already provided in the data making it 0 for all '''

for i in range(len(x_data)):
    x_data[i][2] = 0.00

dataseq = DataSequence(x_data)

fcn(data=dataseq, qbx_thresholds=[1.2], chaining_thr=2, viz=True)

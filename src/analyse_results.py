import os
from os import path
import numpy as np
from matplotlib import pyplot as plt

experiment_name = "experiment_0"
results_path = path.join("results", experiment_name)
map_path = path.join("saved_maps", experiment_name + ".pickle")

data = []
for diffusion_rate in os.listdir(results_path):

    with open(path.join(results_path, diffusion_rate, "index.txt")) as file:
        lines = file.readlines()
        lines = [line.rstrip().split()[1] for line in lines]
    data.append(lines)

data = np.array(data, dtype=np.int32)
x_axis = np.arange(0, 1.01, 0.01)
print(data)

maxes = np.max(data, axis=1)
maxes_best_fit = np.polyval(np.polyfit(x_axis, maxes, 4), x_axis)
mins = np.min(data, axis=1)
mins_best_fit = np.polyval(np.polyfit(x_axis, mins, 4), x_axis)
means = np.mean(data, axis=1)
means_best_fit = np.polyval(np.polyfit(x_axis, means, 4), x_axis)
stds = np.std(data, axis=1)
stds_best_fit = np.polyval(np.polyfit(x_axis, stds, 4), x_axis)



plt.scatter(x_axis, maxes)
plt.plot(x_axis, maxes_best_fit)
plt.show()
plt.scatter(x_axis, mins)
plt.plot(x_axis, mins_best_fit)
plt.show()
plt.scatter(x_axis, means)
plt.plot(x_axis, means_best_fit)
plt.show()
plt.scatter(x_axis, stds)
plt.plot(x_axis, stds_best_fit)
plt.show()


import os
from os import path
import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial import polynomial

experiment_name = "experiment_1"
results_path = path.join("results", experiment_name)
map_path = path.join("saved_maps", experiment_name + ".pickle")

data = []
for diffusion_rate in os.listdir(results_path):
    with open(path.join(results_path, diffusion_rate, "index.txt")) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    data.append(lines[:27])

data = np.array(data, dtype=np.int32)
x_coordinates = np.repeat(np.arange(0, 1.01, 0.01), 27)
y_coordinates = data.flatten()

x_axis = np.arange(0, 1.01, 0.01)

# Fill upper and lower bound
y_polyfit = polynomial.Polynomial.fit(x_coordinates, y_coordinates, 4).convert().coef
y_polyval = polynomial.polyval(x_axis, y_polyfit)
stds = np.std(data, axis=1)
plt.fill_between(x_axis, y_polyval - 2 * stds, y_polyval + 2 * stds)
plt.plot(x_axis, y_polyval, 'red')

# Apply polynomial LSR to upperbound
y_lower_polyfit = polynomial.Polynomial.fit(x_axis, y_polyval - 2 * stds, 4).convert().coef
y_lower_polyval = polynomial.polyval(x_axis, y_lower_polyfit)
plt.plot(x_axis, y_lower_polyval, 'green')

# Apply polynomial LSR to lowerbound
y_upper_polyfit = polynomial.Polynomial.fit(x_axis, y_polyval + 2 * stds, 4).convert().coef
y_upper_polyval = polynomial.polyval(x_axis, y_upper_polyfit)
plt.plot(x_axis, y_upper_polyval, 'green')

plt.xlabel("Pheromone Diffusion Rate")
plt.ylabel("Steps Until Environment Depleted")
plt.show()

# Plot means and polynomial LSR
means = np.mean(data, axis=1)
means_best_fit = np.polyval(np.polyfit(x_axis, means, 4), x_axis)
plt.scatter(x_axis, means)
plt.plot(x_axis, means_best_fit, 'red')
plt.xlabel("Pheromone Diffusion Rate")
plt.ylabel("Steps Until Environment Depleted")
plt.show()


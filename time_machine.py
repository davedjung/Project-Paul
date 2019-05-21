#time_machine.py

import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

dimension_x = 10 #dimensions of container
dimension_y = 10
resolution = 0.1 #resolution of distribution

fig, (ax_box, ax_graph) = plt.subplots(1, 2, num="Maxwell-Boltzmann Distribution Simulation", figsize=[11,5])
ax_box.set_xlim([-dimension_x,dimension_x])
ax_box.set_ylim([-dimension_y,dimension_y])


x_MB = np.arange(0, v_max*2, 0.01)
B = 2 / v2_avg
y_MB = B * x_MB * np.exp(-1 * x_MB**2 / v2_avg) * resolution
x_dist = np.arange(0, v_max*2, resolution)

ax_graph.set_xlim([0, v_max*2])
ax_graph.set_ylim([0, np.amax(y_MB)*1.5])

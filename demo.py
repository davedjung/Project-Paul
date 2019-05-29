#demo.py

#header
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
print("Maxwell-Boltzmann Distribution Simulation Demo")
print("Created by Jung Min Ki")
def mod(v):
    return np.sum(v * v, axis=-1)

#configuration
size = int(input("Number of particles : "))
temperature_scale = float(input("Temperature scale : "))
D = 0.05 #particle diameter
dimension_x = 10 #dimensions of container
dimension_y = 10
tick = 0.1 #time scale for performance mode
age = 0 #generation counter
resolution = 0.1 #resolution of distribution

#initialization
start = timer()
rx = (np.random.random(size) - 0.5) * dimension_x * 1.95
ry = (np.random.random(size) - 0.5) * dimension_y * 1.95
r = np.append(rx, ry)
r.shape = (size, 2)
v = (np.random.random((size,2)) - 0.5) * temperature_scale * 2

#miscellaneous

fig, ax_graph = plt.subplots(1, 1, num="Maxwell-Boltzmann Distribution Simulation Demo")
v_mag = np.sqrt(v[:,0]**2 + v[:,1]**2)
v2_avg = np.mean(v_mag**2)
v_max = np.amax(v_mag)

x_MB = np.arange(0, v_max*2, 0.01)
B = 2 / v2_avg
y_MB = B * x_MB * np.exp(-1 * x_MB**2 / v2_avg) * resolution
x_dist = np.arange(0, v_max*2, resolution)

ax_graph.set_xlim([0, 10])
ax_graph.set_ylim([0, 0.06])

print("setup time: ", timer()-start)

#simulation starts from here...
def refresh(frame):
	global r
	global v
	global age
	print("generation counter: ", age)

	start = timer()
	
	for i in range(size):
		r[i] = r[i] + v[i] * tick
	dists = np.sqrt(mod(r - r[:,np.newaxis]))
	cols2 = (0 < dists) & (dists < D)
	idx_i, idx_j = np.nonzero(cols2)
	for i, j in zip(idx_i, idx_j):
		if j < i:
			continue

		rij = r[i] - r[j]
		d = mod(rij)
		vij = v[i] - v[j]
		dv = np.dot(vij, rij) * rij / d
		v[i] -= dv
		v[j] += dv

		r[i] += tick * v[i]
		r[j] += tick * v[j]

	for i in range(size):
		if r[i][0] >= dimension_x - D/2:
			v[i][0] = -v[i][0]
		elif r[i][0] <= -dimension_x + D/2:
			v[i][0] = -v[i][0]
		if r[i][1] >= dimension_y - D/2:
			v[i][1] = -v[i][1]
		elif r[i][1] <= -dimension_y + D/2:
			v[i][1] = -v[i][1]

	#compute speed distribution
	v_mag = np.sqrt(v[:,0]**2 + v[:,1]**2)
	#v_max = np.amax(v_mag)
	x_dist = np.arange(0,v_max*2, resolution)
	dist = np.histogram(v_mag, bins=x_dist)
	y_dist = dist[0] / size
	y_dist = np.append(np.zeros(1), y_dist)
	print("computation time: ", timer()-start)
	#graphics
	start = timer()
	ax_graph.clear()
	ax_graph.set_xlim([0, 10])
	ax_graph.set_ylim([0, 0.06])
	ax_graph.plot(x_MB,y_MB, 'b')
	width = resolution
	rects = ax_graph.bar(x_dist + resolution*0.1, y_dist, width - resolution*0.2, color='IndianRed')
	print("histogram graphics time: ", timer()-start)
	age += 1
	plt.pause(0.0001)

a = FuncAnimation(fig, refresh, frames=100000, interval=1)
plt.show()

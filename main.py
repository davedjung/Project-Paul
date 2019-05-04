#main.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
from matplotlib.animation import FuncAnimation
from timeit import default_timer as timer

def column(array, i):
	return [row[i] for row in array]

#configuration
size = int(input("Number of particles: "))
D = 0.1 #particle diameter
x_max = 10 #container boundary
x_min = -10
y_max = 10
y_min = -10
tick = 0.1 #time scale
resolution = 120 #resolution of distribution graph
temperature_scale = int(input("Temperature scale: "))

#declarations
r = np.zeros((size,2))
rij = np.zeros(((size,size,2)))
rij_mag2 = np.zeros((size,size))
v = np.zeros((size,2))
v_mag = np.zeros(size)
vij = np.zeros(((size,size,2)))
v2_avg = 0

def mod(v):
    return np.sum(v * v, axis=-1)

#initialization
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/1.1 - (x_max-x_min)/2.2
	r[i,1] = np.random.rand() * (y_max-y_min)/1.1 - (y_max-y_min)/2.2

index = -5
for i in range(size):
	v[i][0] = temperature_scale * index / 3
	v[i][1] = temperature_scale * index / 3
	index = index + 1
	if index == 6:
		index = -5
	v2_avg = v2_avg + v[i][0]**2 + v[i][1]**2
	v_mag[i] = np.sqrt(v[i][0]**2 + v[i][1]**2)

v2_avg = v2_avg / size

fig = plt.figure()
ax = fig.add_axes()
ax = plt.axes()
ax.set_xlim([x_min,x_max])
ax.set_ylim([y_min,y_max])

graph = plt.figure()
ax2 = graph.add_subplot(111)

print("Initial setup complete...")

#simulation starts from here...

def refresh(frame):
	start = timer()
	tb = np.zeros(size)
	container_collision = 0
	global r
	global v
	
#progress-------------------------------------
	for i in range(size):
		r[i] = r[i] + v[i] * tick
	print("Particles moved...")

#check if collision occurs in this tick--------

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
	print("Particle collisions evaluated...")

	for i in range(size):
		if r[i][0] >= x_max - D/2:
			v[i][0] = -v[i][0]
		elif r[i][0] <= x_min + D/2:
			v[i][0] = -v[i][0]
		if r[i][1] >= y_max - D/2:
			v[i][1] = -v[i][1]
		elif r[i][1] <= y_min + D/2:
			v[i][1] = -v[i][1]
	print("Boundary collisions evaluated...")

	

#Maxwell-Boltzmann distribution
	v_max = -1
	v_min = 0
	for i in range(size):
		v_mag[i] = np.sqrt(v[i][0]**2 + v[i][1]**2)
	for i in range(size):
		if v_mag[i] >= v_max:
			v_max = v_mag[i]
	v_dist = np.zeros(resolution)
	v_max = v_max * 1.2
	step = 0.5
	#print("v_max: ", v_max)
	#print("v_mag: ", v_mag)
	for i in range(resolution):
		count = 0
		#print("Lower bound: ", step * i)
		#print("Upper bound: ", step * (i+1))
		for j in range(size):
			if v_mag[j] >= step * i and v_mag[j] < step * (i+1):
				count = count + 1
		#print("Count: ", count)
		v_dist[i] = count
	print("Distribution calculated...")
	#print(v_dist)
	#f= open("distribution.txt","w+")
	#for i in range(resolution):
	#	output = str(v_dist[i])
	#	f.write(output)
	#	f.write(" ")
	#f.close()

	#graphics
	ax.clear()
	ax.set_xlim([x_min,x_max])
	ax.set_ylim([y_min,y_max])
	ax.plot(column(r,0), column(r,1), 'ro', markersize = 3)
	
	x = np.zeros(resolution+1)
	for i in range(resolution):
		x[i+1] = (step * i + step * (i+1)) / 2
	y = np.zeros(resolution+1)
	for i in range(resolution):
		y[i+1] = v_dist[i] / size

	x_boltzmann = np.zeros(10000)
	for i in range(10000):
		x_boltzmann[i] = i * 0.01
	y_boltzmann = np.zeros(10000)
	B = 2 / v2_avg
	for i in range(10000):
		y_boltzmann[i] = B * x_boltzmann[i] * np.exp(-1 * x_boltzmann[i]**2 / v2_avg) * step
	ax2.clear()
	#ax2.set_xlim(0,max(x))
	ax2.set_xlim(0,10)
	ax2.set_ylim(0,max(y_boltzmann)*2)
	#ax2.plot(x, y, 'r', x, y_boltzmann, 'b')
	ax2.plot(x_boltzmann,y_boltzmann, 'b')
	width = step
	rects = ax2.bar(x + step*0.1, y, width - step*0.1, color='IndianRed')
	lifetime = timer() - start
	print("Generation time : " , lifetime)


a = FuncAnimation(fig, refresh, frames=100000, interval=1)
b = FuncAnimation(graph, refresh, frames=100000, interval=1)

plt.show()


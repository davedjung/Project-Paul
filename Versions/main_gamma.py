#main_gamma.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
from matplotlib.animation import FuncAnimation
from timeit import default_timer as timer
from numba import vectorize
import multiprocessing as mp

@vectorize(['float32(float32, float32)'], target='cpu')
def mag_cpu(a, b):
    return a**2 + b**2

def column(array, i):
	return [row[i] for row in array]

print("Maxwell-Boltzmann Distribution Simulation version 0.3")
print("Created by Jung Min Ki")

#configuration
size = int(input("Number of particles: "))
D = 0.5 #particle diameter
x_max = 10 #container boundary
x_min = -10
y_max = 10
y_min = -10
tick = 0.01 #time scale
epoch = 0 #time until next collision
resolution = 10 #resolution of distribution graph
temperature_scale = int(input("Temperature scale: "))

#declarations
r = np.zeros((size,2))
r_next = np.zeros((size,2))
rij = np.zeros(((size,size,2)))
rij_mag2 = np.zeros((size,size))
v = np.zeros((size,2), dtype = np.float32)
v_mag = np.zeros(size, dtype = np.float32)
vij = np.zeros(((size,size,2)))
vij_mag = np.zeros((size,size))
rvij = np.zeros((size,size))
tij = np.zeros((size,size))
tbi = np.zeros(size)
container_collision = 0
x_collision = 0
y_collision = 0
particle_collision = 0
index_1 = 0
index_2 = -1
v2_avg = 0

#initialization
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/1.1 - (x_max-x_min)/2.2
	r[i,1] = np.random.rand() * (y_max-y_min)/1.1 - (y_max-y_min)/2.2

for i in range(size):
    v[i,0] = (np.random.rand() * 2 - 1) * temperature_scale
    v[i,1] = (np.random.rand() * 2 - 1) * temperature_scale
    v2_avg = v2_avg + v[i][0]**2 + v[i][1]**2

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

def collisionDetection(x):
		global tij
		global complexity
		for i in range(x, x+int(size/8)):
			for j in range(size):
				global tij
				if i<=j:
					break
				else:
					rij[i][j] = r_next[i] - r_next[j]
					if rij[i][j][0]**2 + rij[i][j][1]**2 <= D**2:
						complexity += 1
						rij[i][j] = r[i] - r[j]
						rij_mag2[i][j] = rij[i][j][0]**2 + rij[i][j][1]**2
						vij[i][j] = v[i] - v[j]
						vij_mag[i][j] = np.sqrt(vij[i][j][0]**2 + vij[i][j][1]**2)
						rvij[i][j] = np.dot(rij[i][j], vij[i][j])
						b2 = rij_mag2[i][j] - (rvij[i][j] / vij_mag[i][j])**2
						tij[i][j] = -1/vij_mag[i][j] * (rvij[i][j]/vij_mag[i][j] + (D**2 - b2)**(1/2))
		

def refresh(frame):
	global tij
	start = timer()
	tij = np.zeros((size,size))
	tb = np.zeros(size)
	container_collision = 0
	particle_collision = 0
	rij = np.zeros(((size,size,2)))
	global r
	global r_next
	global v
	
#progress-------------------------------------
	for i in range(size):
		r_next[i] = r[i] + v[i] * tick
	print("Particles moved...")
#check if collision occurs in this tick--------
	global complexity
	complexity = 0
	for i in range(size):
			for j in range(size):
				if i<=j:
					break
				else:
					rij[i][j] = r_next[i] - r_next[j]
					if rij[i][j][0]**2 + rij[i][j][1]**2 <= D**2:
						complexity += 1
						rij[i][j] = r[i] - r[j]
						rij_mag2[i][j] = rij[i][j][0]**2 + rij[i][j][1]**2
						vij[i][j] = v[i] - v[j]
						vij_mag[i][j] = np.sqrt(vij[i][j][0]**2 + vij[i][j][1]**2)
						rvij[i][j] = np.dot(rij[i][j], vij[i][j])
						b2 = rij_mag2[i][j] - (rvij[i][j] / vij_mag[i][j])**2
						tij[i][j] = -1/vij_mag[i][j] * (rvij[i][j]/vij_mag[i][j] + (D**2 - b2)**(1/2))
	print("Particle collisions evaluated...")
	print("Complexity: ", complexity)
	for i in range(size):
		tx = 1.7976931348623157e+308
		ty = 1.7976931348623157e+308
		if r_next[i][0] >= x_max - D/2:
			tx = (x_max - D/2 - r[i][0]) / v[i][0]
		elif r_next[i][0] <= x_min + D/2:
			tx = (r[i][0] - x_min - D/2) / v[i][0]
		if r_next[i][1] >= y_max - D/2:
			ty = (y_max - D/2 - r[i][1]) / v[i][1]
		elif r_next[i][1] <= y_min + D/2:
			ty = (r[i][1] - y_min - D/2) / v[i][1]
		tb[i] = min(abs(tx), abs(ty))
		if tb[i] == 1.7976931348623157e+308:
			tb[i] = 0
	print("Boundary collisions evaluated...")

	epoch = 1.7976931348623157e+308-1
	for i in range(size):
		if tb[i] > 0 and epoch >= tb[i]:
			container_collision = 1
			index_1 = i
			epoch = tb[i]

	for i in range(size):
		for j in range(size):
			if tij[i][j] > 0 and epoch >= tij[i][j]:
				particle_collision = 1
				container_collision = 0
				index_1 = i
				index_2 = j
				epoch = tij[i][j]
	print("Closest collision found...")

	if epoch >= tick:
		for i in range(size):
			r[i] = r_next[i]
	else:
		for i in range(size):
			r[i] = r[i] + v[i] * epoch
		if container_collision == 1:
			if r[index_1][0] >= x_max - D/2 or r[index_1][0] <= x_min + D/2:
				v[index_1][0] = -v[index_1][0]
			if r[index_1][1] >= y_max - D/2 or r[index_1][1] <= y_min + D/2:
				v[index_1][1] = -v[index_1][1]
		else:
			delta_v = -rvij[index_1][index_2]/(rij_mag2[index_1][index_2]) * rij[index_1][index_2]
			v[index_1] = v[index_1] + delta_v
			v[index_2] = v[index_2] - delta_v
	print("Collision handled...")

	lifetime = timer() - start
	print("Generation time : " , lifetime)

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
	step = (v_max - v_min) / resolution
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
	print(v_dist)
	f= open("distribution.txt","w+")
	for i in range(resolution):
		output = str(v_dist[i])
		f.write(output)
		f.write(" ")
	f.close()

	#graphics
	ax.clear()
	ax.set_xlim([x_min,x_max])
	ax.set_ylim([y_min,y_max])
	ax.plot(column(r,0), column(r,1), 'ro')

	x = np.zeros(resolution+1)
	for i in range(resolution):
		x[i+1] = (step * i + step * (i+1)) / 2
	y = np.zeros(resolution+1)
	for i in range(resolution):
		y[i+1] = v_dist[i] / size

	y_boltzmann = np.zeros(resolution+1)
	B = 2 / v2_avg
	for i in range(resolution+1):
		y_boltzmann[i] = B * x[i] * np.exp(-1 * x[i]**2 / v2_avg) * step
	ax2.clear()
	ax2.set_xlim(0,max(x))
	ax2.set_ylim(0,max(y)*1.1)
	ax2.plot(x, y, 'r', x, y_boltzmann, 'b')
	#ax2.plot(x,y)

a = FuncAnimation(fig, refresh, frames=100000, interval=1)
b = FuncAnimation(graph, refresh, frames=100000, interval=1)
plt.show()

#main.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
from matplotlib.animation import FuncAnimation
from timeit import default_timer as timer
from numba import vectorize

@vectorize(['float32(float32, float32)'], target='cpu')
def mag_cpu(a, b):
    return a**2 + b**2

def column(array, i):
	return [row[i] for row in array]

#configuration
size = int(input("Number of particles: "))
D = 0.5 #particle diameter
x_max = 10 #container boundary
x_min = -10
y_max = 10
y_min = -10
tick = 0.1 #time scale
epoch = 0 #time until next collision
#resolution = 20 #resolution of distribution graph

#declarations

r = np.zeros((size,2))
r_next = np.zeros((size,2))
rij = np.zeros(((size,size,2)))
rij_mag = np.zeros((size,size))
r2ij = np.zeros((size,size))
v = np.zeros((size,2))
v_mag = np.zeros(size)
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

#initialization
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

for i in range(size):
    v[i,0] = np.random.rand()*10 - 5
    v[i,1] = np.random.rand()*10 - 5

fig = plt.figure()
ax = fig.add_axes()
ax = plt.axes()
ax.set_xlim([x_min,x_max])
ax.set_ylim([y_min,y_max])

print("Initial setup complete...")

#simulation starts from here...


def refresh(frame):
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
	for i in range(size):
		for j in range(size):
			if i<=j:
				break
			else:
				rij[i][j] = r_next[i] - r_next[j]
				if rij[i][j][0]**2 + rij[i][j][1]**2 <= D**2:
					rij[i][j] = r[i] - r[j]
					rij_mag[i][j] = np.sqrt(rij[i][j][0]**2 + rij[i][j][1]**2)
					vij[i][j] = v[i] - v[j]
					vij_mag[i][j] = np.sqrt(vij[i][j][0]**2 + vij[i][j][1]**2)
					rvij[i][j] = np.dot(rij[i][j], vij[i][j])
					b2 = rij_mag[i][j] ** 2 - (rvij[i][j] / vij_mag[i][j])**2
					tij[i][j] = -1/vij_mag[i][j] * (rvij[i][j]/vij_mag[i][j] + (D**2 - b2**2)**(1/2))
	print("Particle collisions evaluated...")

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
		tb[i] = min(tx, ty)
		if tb[i] == 1.7976931348623157e+308:
			tb[i] = 0
	print("Boundary collisions evaluated...")

	print(tb)

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
			print("Particle collision!!!!!!!!!!!!!!!!!!!!!!!!!")
			delta_v = -rvij[index_1][index_2]/rij_mag[index_1][index_2] * rij[index_1][index_2]
			v[index_1] = v[index_1] + delta_v
			v[index_2] = v[index_2] - delta_v
	print("Collision handled...")

#reset tij, r', v_next
#progess r->r'
#check overlapping: determine tij (somehow)
#	we know i, j, tick
#	use r, v to calculate tij???
#find the lowest tij
#if lowest = 0 (no collision):
#	r = r'
#else:
#	rewind!!!
#	progress r->r' for epoch = tij_lowest
#	compute v_next for i,j
#	update v_next -> v

	#graphics
	ax.clear()
	ax.set_xlim([x_min,x_max])
	ax.set_ylim([y_min,y_max])
	ax.plot(column(r,0), column(r,1), 'ro')

a = FuncAnimation(fig, refresh, frames=100000, interval=10)

plt.show()

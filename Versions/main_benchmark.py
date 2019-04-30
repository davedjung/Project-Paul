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
D = 0.5 #particle diameter
x_max = 10 #container boundary
x_min = -10
y_max = 10
y_min = -10
tick = 0.01 #time scale
epoch = 0 #time until next collision
resolution = 20 #resolution of distribution graph
generation = 0

#declarations
#time_absolute = 0 #global timer
r = np.zeros((size,2))
rij = np.zeros(((size,size,2)))
rij_mag = np.zeros((size,size))
v = np.zeros((size,2))
v_mag = np.zeros(size)
vij = np.zeros(((size,size,2)))
vij_mag = np.zeros((size,size))
rvij = np.zeros((size,size))
b2ij = np.zeros((size,size))
tij = np.zeros((size,size))
tbi = np.zeros(size)
container_collision = 1 #collision with container
v_next = np.zeros((size,2)) #new velocity for next epoch

#initialization
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

for i in range(size):
    v[i,0] = np.random.rand()*10 - 5
    v[i,1] = np.random.rand()*10 - 5
#simulation starts from here...

print("Initial setup complete...")

fig = plt.figure()
ax = fig.add_axes()
ax = plt.axes()
ax.set_xlim([x_min,x_max])
ax.set_ylim([y_min,y_max])

def refresh(frame):
	global generation
	print("Generation #: ", generation)
#reset time-sensitive parameters--------------
	tij = np.zeros((size,size))
	tbi = np.zeros(size)

#compute relevant parameters------------------
	print("Start computing...")
	start = timer()
	for i in range(size):
		for j in range(size):
			if j < i:
				rij[i][j] = - rij[j][i]
				rij_mag[i][j] = rij_mag[j][i]
			else:
				rij[i][j] = r[i] - r[j]
				rij_mag[i][j] = np.sqrt(np.square(rij[i][j][0])+np.square(rij[i][j][1]))
	lifetime = timer() - start
	print("Step 1 complete : " , lifetime)
	start = timer()
	for i in range(size):
		v_mag[i] = np.sqrt(np.square(v[i][0])+np.square(v[i][1]))
	lifetime = timer() - start
	print("Step 2 complete : " , lifetime)
	start = timer()
	for i in range(size):
		for j in range(size):
			if j < i:
				vij[i][j] = - vij[j][i]
				vij_mag[i][j] = vij_mag[j][i]
			else:
				vij[i][j] = v[i] - v[j]
				vij_mag[i][j] = np.sqrt(np.square(vij[i][j][0])+np.square(vij[i][j][1]))
	lifetime = timer() - start
	print("Step 3 complete : " , lifetime)
	start = timer()
	for i in range(size):
		for j in range(size):
			if j < i:
				rvij[i][j] = rvij[j][i]
			else:
				rvij[i][j] = np.dot(rij[i][j], vij[i][j])
	lifetime = timer() - start
	print("Step 4 complete : " , lifetime)
	start = timer()
	for i in range(size):
		for j in range(size):
			if i!=j:
				if j < i:
					b2ij[i][j] = b2ij[j][i]
				else:
					b2ij[i][j] = rij_mag[i][j] ** 2 - (rvij[i][j] / vij_mag[i][j])**2
	lifetime = timer() - start
	print("Step 5 complete : " , lifetime)
	print("Parameters computed...")

#handle collisions--------------------------
	#collision time between particles
	for i in range(size):
		for j in range(size):
			if i!=j:
				if rvij[i][j]<0:
					if b2ij[i][j] <= D**2:
						tij[i][j] = -1/vij_mag[i][j] * (rvij[i][j]/vij_mag[i][j] + (D**2 - b2ij[i][j]**2)**(1/2))
	
	print("Collision between particles evaluated...")

	#find closest collision
	epoch = 1.7976931348623157e+308
	index_1 = 0
	index_2 = -1
	container_collision = 0

	for i in range(size):
		for j in range(size):
			if epoch >= tij[i][j] and tij[i][j]>0:
				index_1 = i
				index_2 = j
				epoch = tij[i][j]
				container_collision = 0

	print("Closest collision found... @", epoch)

	#collision time with the boundary
	x_collision = 0
	y_collision = 0
	for i in range(size):
		if v[i][0]>=0:
			tx=((x_max-D/2)-r[i][0])/v[i][0]
		else:
			tx=((x_min+D/2)-r[i][0])/v[i][0]
		if v[i][1]>=0:
			ty=((y_max-D/2)-r[i][1])/v[i][1]
		else:
			ty=((y_min+D/2)-r[i][1])/v[i][1]
		tbi[i] = min(tx,ty)
		if epoch >= tbi[i]:
			index_1 = i
			index_2 = -1
			epoch = tbi[i]
			container_collision = 1
			if epoch == tx:
				x_collision = 1
				y_collision = 0
			else:
				x_collision = 0
				y_collision = 1

	print("Collision with boundary evaluated...")

#compute next epoch's velocity profile
	if container_collision == 1:
		print("Collision with boundary.... @", epoch )
		for i in range(size):
			v_next[i] = v[i]
			if i == index_1:
				if x_collision == 1:
					v_next[i][0] = -v_next[i][0]
				else:
					v_next[i][1] = -v_next[i][1]
	else:
		for i in range(size):
			v_next[i] = v[i]
			if i == index_1:
				delta_v = -rvij[index_1][index_2]/rij_mag[index_1][index_2] * rij[index_1][index_2]
				v_next[i] = v_next[i] + delta_v
			elif i == index_2:
				delta_v = rvij[index_1][index_2]/rij_mag[index_1][index_2] * rij[index_1][index_2]
				v_next[i] = v_next[i] + delta_v

	print("Next epoch's velocity profile calculated...")

#progress-------------------------------------
	def progress(time):
		for i in range(size):
			r[i] = r[i] + v[i] * time

#update velocity due to collision--------------
	def update():
		for i in range(size):
			v[i] = v_next[i]

#check if collision occurs in this tick--------
	if tick < epoch:
		progress(tick)
		#time_absolute = time_absolute + tick
	else:
		progress(epoch)
		update()
		#time_absolute = time_absolute + epoch

	print("Particles moved...")

#debugging:
	OB = 0
	for i in range(size):
		if r[i][0] >= x_max or r[i][0] <= x_min:
			OB = 1
		elif r[i][1] >= y_max or r[i][1] <= y_min:
			OB = 1
	if OB == 1:
		print("OB!!!!!!!!!!!")

#Maxwell-Boltzmann distribution
	v_max = -1
	v_min = 0
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

	generation = generation + 1

	#graphics
	ax.clear()
	ax.set_xlim([x_min,x_max])
	ax.set_ylim([y_min,y_max])
	ax.plot(column(r,0), column(r,1), 'ro')

a = FuncAnimation(fig, refresh, frames=100000)

plt.show()

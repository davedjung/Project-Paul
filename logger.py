#logger.py

#header
import numpy as np
from timeit import default_timer as timer
print("Maxwell-Boltzmann Distribution Simulation version 0.5")
print("Created by Jung Min Ki")
def mod(v):
    return np.sum(v * v, axis=-1)

#configuration
size = int(input("Number of particles : "))
temperature_scale = float(input("Temperature scale : "))
mode = int(input("Precision mode [0] or Performance mode [1] ? (Default: Precision) : "))
if mode == 1:
	mode = 1
else:
	mode = 0
D = 0.01 #particle diameter
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

v_mag = np.sqrt(v[:,0]**2 + v[:,1]**2)
v2_avg = np.mean(v_mag**2)
v_max = np.amax(v_mag)

x_MB = np.arange(0, v_max*2, 0.01)
B = 2 / v2_avg
y_MB = B * x_MB * np.exp(-1 * x_MB**2 / v2_avg) * resolution
x_dist = np.arange(0, v_max*2, resolution)

f= open("distribution.txt","a")
f.write(str(v_max))
f.write(" ")
f.write(str(v2_avg))
f.write("\n")

print("setup time: ", timer()-start)

#simulation starts from here...
while True:

	print("generation counter: ", age)

	start = timer()
	
	if mode == 1:
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

	else:
		#collision detection
		rij = r.reshape(size,1,2) - r
		r2ij = (rij ** 2).sum(2)
		
		vij = v.reshape(size,1,2) - v
		v2ij = (vij ** 2).sum(2)
		vij_mag = np.sqrt(vij[:,:,0]**2 + vij[:,:,1]**2)

		rvij = rij[:,:,0]*vij[:,:,0] + rij[:,:,1]*vij[:,:,1]
		v2ij[v2ij<=0] = np.inf
		b2ij = r2ij - rvij / v2ij
		vij_mag[vij_mag<=0] = np.inf
		b2ij[b2ij>D**2] = 0
		tij = -1 / vij_mag * (rvij/vij_mag + np.sqrt(D**2 - b2ij))
		tij[tij<=0] = np.inf
		index = np.arange(size)
		tij[index, index] = np.inf
		
		tc = np.amin(tij)
		
		index = np.unravel_index(np.argmin(tij), tij.shape)
		
		#boundary collision
		tx_max = ((dimension_x - D/2) - r[:,0]) / v[:,0]
		tx_min = ((-dimension_x + D/2) - r[:,0]) / v[:,0]
		ty_max = ((dimension_y - D/2) - r[:,1]) / v[:,1]
		ty_min = ((-dimension_y + D/2) - r[:,1]) / v[:,1]
		
		tx_max[tx_max<=0] = np.inf
		tx_min[tx_min<=0] = np.inf
		ty_max[ty_max<=0] = np.inf
		ty_min[ty_min<=0] = np.inf
		
		tx = min(np.amin(tx_max), np.amin(tx_min))
		if tx == np.amin(tx_max):
			index_bx = np.argmin(tx_max)
		else:
			index_bx = np.argmin(tx_min)
		ty = min(np.amin(ty_max), np.amin(ty_min))
		if ty == np.amin(ty_max):
			index_by = np.argmin(ty_max)
		else:
			index_by = np.argmin(ty_min)
		
		epoch = min(tc, tx, ty)
		
		#progress
		r = r + v * epoch

		#update velocity profile
		if epoch==tc:
			delta_v = -rvij[index[0]][index[1]]/(r2ij[index[0]][index[1]]) * rij[index[0]][index[1]]
			v[index[0]] += delta_v
			v[index[1]] -= delta_v
		elif epoch==tx:
			v[index_bx][0] *= -1
		else:
			v[index_by][1] *= -1

		print("epoch: ", epoch)
	
	#compute speed distribution
	v_mag = np.sqrt(v[:,0]**2 + v[:,1]**2)
	#v_max = np.amax(v_mag)
	x_dist = np.arange(0,v_max*2, resolution)
	dist = np.histogram(v_mag, bins=x_dist)
	y_dist = dist[0] / size
	y_dist = np.append(np.zeros(1), y_dist)
	print("time: ", timer()-start)
	

	output = str(y_dist)
	f.write(output)
	f.write(" ")

	age += 1
f.close()

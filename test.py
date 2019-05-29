from multiprocessing import Pool
import numpy as np
from timeit import default_timer as timer


size = 10000

rx = (np.random.random(size) - 0.5) * 10 * 1.95
ry = (np.random.random(size) - 0.5) * 10 * 1.95
r = np.append(rx, ry)
r.shape = (size, 2)

start = timer()
rij = r.reshape(size,1,2) - r
r2ij = (rij ** 2).sum(2)
rij_mag = np.sqrt(r2ij)
print("No parallelization: ", timer()-start)

rx = (np.random.random(size) - 0.5) * 10 * 1.95
ry = (np.random.random(size) - 0.5) * 10 * 1.95
r = np.append(rx, ry)
r.shape = (size, 2)

rij_mag = np.zeros((size,size))

def compute(i,j):
	global r
	global rij_mag
	rij_mag = np.sqrt((r[i][0]-r[j][0])**2 + (r[i][1]-r[j][1])**2)

vector_x = np.arange(size)
vector_y = np.arange(size)

if __name__ == '__main__':
	start = timer()
	pool = Pool()
	roots = pool.starmap(compute, zip(vector_x, vector_y))
	print("Parallelization: ", timer()-start)

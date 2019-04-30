#Vectorization_Benchmark.py

from timeit import default_timer as timer
from numba import vectorize
import numpy as np

print("Magnitude Computation")

@vectorize(['float32(float32, float32)'], target='cpu')
def mag_cpu(a, b):
    return a**2 + b**2

@vectorize(['float32(float32, float32)'], target='cuda')
def mag_gpu(a, b):
    return a**2 + b**2

print("Set 1: size = 10")

#declarations
size = 10
r = np.zeros((size,2))
rij = np.zeros(((size,size,2)))
rij_mag = np.zeros((size,size))
rij0 = np.zeros((size,size), dtype=np.float32)
rij1 = np.zeros((size,size), dtype=np.float32)
x_max = 10
x_min = -10
y_max = 10
y_min = -10

#initialization
r = np.zeros((size,2))
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

#computation
start = timer()
for i in range(size):
	for j in range(size):
		rij[i][j] = r[i] - r[j]
		rij_mag[i][j] = np.sqrt(np.square(rij[i][j][0])+np.square(rij[i][j][1]))
lifetime = timer() - start
print("Test 1 complete [linear] : " , lifetime)

#initialization
r = np.zeros((size,2))
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

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
print("Test 2 complete [linear optimzed] : " , lifetime)

#initialization
r = np.zeros((size,2),dtype=np.float32)
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

start = timer()
for i in range(size):
	for j in range(size):
		rij0[i][j] = r[i][0] - r[j][0]
		rij1[i][j] = r[i][1] - r[j][1]
for i in range(size):
	rij_mag[i] = mag_cpu(rij0[i], rij1[i])
	rij_mag = np.sqrt(rij_mag)
lifetime = timer() - start
print("Test 3 complete [vectorized] : " , lifetime)

#initialization
r = np.zeros((size,2),dtype=np.float32)
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

start = timer()
for i in range(size):
	for j in range(size):
		rij0[i][j] = r[i][0] - r[j][0]
		rij1[i][j] = r[i][1] - r[j][1]
for i in range(size):
	rij_mag[i] = mag_gpu(rij0[i], rij1[i])
	rij_mag = np.sqrt(rij_mag)
lifetime = timer() - start
print("Test 4 complete [gpu] : " , lifetime)

print("Set 2: size = 20")

#declarations
size = 20
r = np.zeros((size,2))
rij = np.zeros(((size,size,2)))
rij_mag = np.zeros((size,size))
rij0 = np.zeros((size,size), dtype=np.float32)
rij1 = np.zeros((size,size), dtype=np.float32)
x_max = 10
x_min = -10
y_max = 10
y_min = -10

#initialization
r = np.zeros((size,2))
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

#computation
start = timer()
for i in range(size):
	for j in range(size):
		rij[i][j] = r[i] - r[j]
		rij_mag[i][j] = np.sqrt(np.square(rij[i][j][0])+np.square(rij[i][j][1]))
lifetime = timer() - start
print("Test 1 complete [linear] : " , lifetime)

#initialization
r = np.zeros((size,2))
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

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
print("Test 2 complete [linear optimzed] : " , lifetime)

#initialization
r = np.zeros((size,2),dtype=np.float32)
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

start = timer()
for i in range(size):
	for j in range(size):
		rij0[i][j] = r[i][0] - r[j][0]
		rij1[i][j] = r[i][1] - r[j][1]
for i in range(size):
	rij_mag[i] = mag_cpu(rij0[i], rij1[i])
	rij_mag = np.sqrt(rij_mag)
lifetime = timer() - start
print("Test 3 complete [vectorized] : " , lifetime)

#initialization
r = np.zeros((size,2),dtype=np.float32)
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

start = timer()
for i in range(size):
	for j in range(size):
		rij0[i][j] = r[i][0] - r[j][0]
		rij1[i][j] = r[i][1] - r[j][1]
for i in range(size):
	rij_mag[i] = mag_gpu(rij0[i], rij1[i])
	rij_mag = np.sqrt(rij_mag)
lifetime = timer() - start
print("Test 4 complete [gpu] : " , lifetime)

print("Set 3: size = 50")

#declarations
size = 50
r = np.zeros((size,2))
rij = np.zeros(((size,size,2)))
rij_mag = np.zeros((size,size))
rij0 = np.zeros((size,size), dtype=np.float32)
rij1 = np.zeros((size,size), dtype=np.float32)
x_max = 10
x_min = -10
y_max = 10
y_min = -10

#initialization
r = np.zeros((size,2))
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

#computation
start = timer()
for i in range(size):
	for j in range(size):
		rij[i][j] = r[i] - r[j]
		rij_mag[i][j] = np.sqrt(np.square(rij[i][j][0])+np.square(rij[i][j][1]))
lifetime = timer() - start
print("Test 1 complete [linear] : " , lifetime)

#initialization
r = np.zeros((size,2))
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

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
print("Test 2 complete [linear optimzed] : " , lifetime)

#initialization
r = np.zeros((size,2),dtype=np.float32)
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

start = timer()
for i in range(size):
	for j in range(size):
		rij0[i][j] = r[i][0] - r[j][0]
		rij1[i][j] = r[i][1] - r[j][1]
for i in range(size):
	rij_mag[i] = mag_cpu(rij0[i], rij1[i])
	rij_mag = np.sqrt(rij_mag)
lifetime = timer() - start
print("Test 3 complete [vectorized] : " , lifetime)

#initialization
r = np.zeros((size,2),dtype=np.float32)
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

start = timer()
for i in range(size):
	for j in range(size):
		rij0[i][j] = r[i][0] - r[j][0]
		rij1[i][j] = r[i][1] - r[j][1]
for i in range(size):
	rij_mag[i] = mag_gpu(rij0[i], rij1[i])
	rij_mag = np.sqrt(rij_mag)
lifetime = timer() - start
print("Test 4 complete [gpu] : " , lifetime)

print("Set 4: size = 100")

#declarations
size = 100
r = np.zeros((size,2))
rij = np.zeros(((size,size,2)))
rij_mag = np.zeros((size,size))
rij0 = np.zeros((size,size), dtype=np.float32)
rij1 = np.zeros((size,size), dtype=np.float32)
x_max = 10
x_min = -10
y_max = 10
y_min = -10

#initialization
r = np.zeros((size,2))
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

#computation
start = timer()
for i in range(size):
	for j in range(size):
		rij[i][j] = r[i] - r[j]
		rij_mag[i][j] = np.sqrt(np.square(rij[i][j][0])+np.square(rij[i][j][1]))
lifetime = timer() - start
print("Test 1 complete [linear] : " , lifetime)

#initialization
r = np.zeros((size,2))
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

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
print("Test 2 complete [linear optimzed] : " , lifetime)

#initialization
r = np.zeros((size,2),dtype=np.float32)
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

start = timer()
for i in range(size):
	for j in range(size):
		rij0[i][j] = r[i][0] - r[j][0]
		rij1[i][j] = r[i][1] - r[j][1]
for i in range(size):
	rij_mag[i] = mag_cpu(rij0[i], rij1[i])
	rij_mag = np.sqrt(rij_mag)
lifetime = timer() - start
print("Test 3 complete [vectorized] : " , lifetime)

#initialization
r = np.zeros((size,2),dtype=np.float32)
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

start = timer()
for i in range(size):
	for j in range(size):
		rij0[i][j] = r[i][0] - r[j][0]
		rij1[i][j] = r[i][1] - r[j][1]
for i in range(size):
	rij_mag[i] = mag_gpu(rij0[i], rij1[i])
	rij_mag = np.sqrt(rij_mag)
lifetime = timer() - start
print("Test 4 complete [gpu] : " , lifetime)

print("Set 5: size = 200")

#declarations
size = 200
r = np.zeros((size,2))
rij = np.zeros(((size,size,2)))
rij_mag = np.zeros((size,size))
rij0 = np.zeros((size,size), dtype=np.float32)
rij1 = np.zeros((size,size), dtype=np.float32)
x_max = 10
x_min = -10
y_max = 10
y_min = -10

#initialization
r = np.zeros((size,2))
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

#computation
start = timer()
for i in range(size):
	for j in range(size):
		rij[i][j] = r[i] - r[j]
		rij_mag[i][j] = np.sqrt(np.square(rij[i][j][0])+np.square(rij[i][j][1]))
lifetime = timer() - start
print("Test 1 complete [linear] : " , lifetime)

#initialization
r = np.zeros((size,2))
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

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
print("Test 2 complete [linear optimzed] : " , lifetime)

#initialization
r = np.zeros((size,2),dtype=np.float32)
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

start = timer()
for i in range(size):
	for j in range(size):
		rij0[i][j] = r[i][0] - r[j][0]
		rij1[i][j] = r[i][1] - r[j][1]
for i in range(size):
	rij_mag[i] = mag_cpu(rij0[i], rij1[i])
	rij_mag = np.sqrt(rij_mag)
lifetime = timer() - start
print("Test 3 complete [vectorized] : " , lifetime)

#initialization
r = np.zeros((size,2),dtype=np.float32)
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

start = timer()
for i in range(size):
	for j in range(size):
		rij0[i][j] = r[i][0] - r[j][0]
		rij1[i][j] = r[i][1] - r[j][1]
for i in range(size):
	rij_mag[i] = mag_gpu(rij0[i], rij1[i])
	rij_mag = np.sqrt(rij_mag)
lifetime = timer() - start
print("Test 4 complete [gpu] : " , lifetime)

print("Set 6: size = 500")

#declarations
size = 500
r = np.zeros((size,2))
rij = np.zeros(((size,size,2)))
rij_mag = np.zeros((size,size))
rij0 = np.zeros((size,size), dtype=np.float32)
rij1 = np.zeros((size,size), dtype=np.float32)
x_max = 10
x_min = -10
y_max = 10
y_min = -10

#initialization
r = np.zeros((size,2))
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

#computation
start = timer()
for i in range(size):
	for j in range(size):
		rij[i][j] = r[i] - r[j]
		rij_mag[i][j] = np.sqrt(np.square(rij[i][j][0])+np.square(rij[i][j][1]))
lifetime = timer() - start
print("Test 1 complete [linear] : " , lifetime)

#initialization
r = np.zeros((size,2))
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

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
print("Test 2 complete [linear optimzed] : " , lifetime)

#initialization
r = np.zeros((size,2),dtype=np.float32)
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

start = timer()
for i in range(size):
	for j in range(size):
		rij0[i][j] = r[i][0] - r[j][0]
		rij1[i][j] = r[i][1] - r[j][1]
for i in range(size):
	rij_mag[i] = mag_cpu(rij0[i], rij1[i])
	rij_mag = np.sqrt(rij_mag)
lifetime = timer() - start
print("Test 3 complete [vectorized] : " , lifetime)

#initialization
r = np.zeros((size,2),dtype=np.float32)
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

start = timer()
for i in range(size):
	for j in range(size):
		rij0[i][j] = r[i][0] - r[j][0]
		rij1[i][j] = r[i][1] - r[j][1]
for i in range(size):
	rij_mag[i] = mag_gpu(rij0[i], rij1[i])
	rij_mag = np.sqrt(rij_mag)
lifetime = timer() - start
print("Test 4 complete [gpu] : " , lifetime)

print("Set 7: size = 1000")

#declarations
size = 1000
r = np.zeros((size,2))
rij = np.zeros(((size,size,2)))
rij_mag = np.zeros((size,size))
rij0 = np.zeros((size,size), dtype=np.float32)
rij1 = np.zeros((size,size), dtype=np.float32)
x_max = 10
x_min = -10
y_max = 10
y_min = -10

#initialization
r = np.zeros((size,2))
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

#computation
start = timer()
for i in range(size):
	for j in range(size):
		rij[i][j] = r[i] - r[j]
		rij_mag[i][j] = np.sqrt(np.square(rij[i][j][0])+np.square(rij[i][j][1]))
lifetime = timer() - start
print("Test 1 complete [linear] : " , lifetime)

#initialization
r = np.zeros((size,2))
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

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
print("Test 2 complete [linear optimzed] : " , lifetime)

#initialization
r = np.zeros((size,2),dtype=np.float32)
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

start = timer()
for i in range(size):
	for j in range(size):
		rij0[i][j] = r[i][0] - r[j][0]
		rij1[i][j] = r[i][1] - r[j][1]
for i in range(size):
	rij_mag[i] = mag_cpu(rij0[i], rij1[i])
	rij_mag = np.sqrt(rij_mag)
lifetime = timer() - start
print("Test 3 complete [vectorized] : " , lifetime)

#initialization
r = np.zeros((size,2),dtype=np.float32)
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

start = timer()
for i in range(size):
	for j in range(size):
		rij0[i][j] = r[i][0] - r[j][0]
		rij1[i][j] = r[i][1] - r[j][1]
for i in range(size):
	rij_mag[i] = mag_gpu(rij0[i], rij1[i])
	rij_mag = np.sqrt(rij_mag)
lifetime = timer() - start
print("Test 4 complete [gpu] : " , lifetime)

print("Set 8: size = 2000")

#declarations
size = 2000
r = np.zeros((size,2))
rij = np.zeros(((size,size,2)))
rij_mag = np.zeros((size,size))
rij0 = np.zeros((size,size), dtype=np.float32)
rij1 = np.zeros((size,size), dtype=np.float32)
x_max = 10
x_min = -10
y_max = 10
y_min = -10

#initialization
r = np.zeros((size,2))
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

#computation
start = timer()
for i in range(size):
	for j in range(size):
		rij[i][j] = r[i] - r[j]
		rij_mag[i][j] = np.sqrt(np.square(rij[i][j][0])+np.square(rij[i][j][1]))
lifetime = timer() - start
print("Test 1 complete [linear] : " , lifetime)

#initialization
r = np.zeros((size,2))
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

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
print("Test 2 complete [linear optimzed] : " , lifetime)

#initialization
r = np.zeros((size,2),dtype=np.float32)
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

start = timer()
for i in range(size):
	for j in range(size):
		rij0[i][j] = r[i][0] - r[j][0]
		rij1[i][j] = r[i][1] - r[j][1]
for i in range(size):
	rij_mag[i] = mag_cpu(rij0[i], rij1[i])
	rij_mag = np.sqrt(rij_mag)
lifetime = timer() - start
print("Test 3 complete [vectorized] : " , lifetime)

#initialization
r = np.zeros((size,2),dtype=np.float32)
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

start = timer()
for i in range(size):
	for j in range(size):
		rij0[i][j] = r[i][0] - r[j][0]
		rij1[i][j] = r[i][1] - r[j][1]
for i in range(size):
	rij_mag[i] = mag_gpu(rij0[i], rij1[i])
	rij_mag = np.sqrt(rij_mag)
lifetime = timer() - start
print("Test 4 complete [gpu] : " , lifetime)

print("Set 9: size = 3000")

#declarations
size = 3000
r = np.zeros((size,2))
rij = np.zeros(((size,size,2)))
rij_mag = np.zeros((size,size))
rij0 = np.zeros((size,size), dtype=np.float32)
rij1 = np.zeros((size,size), dtype=np.float32)
x_max = 10
x_min = -10
y_max = 10
y_min = -10

#initialization
r = np.zeros((size,2))
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

#computation
start = timer()
for i in range(size):
	for j in range(size):
		rij[i][j] = r[i] - r[j]
		rij_mag[i][j] = np.sqrt(np.square(rij[i][j][0])+np.square(rij[i][j][1]))
lifetime = timer() - start
print("Test 1 complete [linear] : " , lifetime)

#initialization
r = np.zeros((size,2))
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

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
print("Test 2 complete [linear optimzed] : " , lifetime)

#initialization
r = np.zeros((size,2),dtype=np.float32)
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

start = timer()
for i in range(size):
	for j in range(size):
		rij0[i][j] = r[i][0] - r[j][0]
		rij1[i][j] = r[i][1] - r[j][1]
for i in range(size):
	rij_mag[i] = mag_cpu(rij0[i], rij1[i])
	rij_mag = np.sqrt(rij_mag)
lifetime = timer() - start
print("Test 3 complete [vectorized] : " , lifetime)

#initialization
r = np.zeros((size,2),dtype=np.float32)
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

start = timer()
for i in range(size):
	for j in range(size):
		rij0[i][j] = r[i][0] - r[j][0]
		rij1[i][j] = r[i][1] - r[j][1]
for i in range(size):
	rij_mag[i] = mag_gpu(rij0[i], rij1[i])
	rij_mag = np.sqrt(rij_mag)
lifetime = timer() - start
print("Test 4 complete [gpu] : " , lifetime)

print("Set 10: size = 4000")

#declarations
size = 4000
r = np.zeros((size,2))
rij = np.zeros(((size,size,2)))
rij_mag = np.zeros((size,size))
rij0 = np.zeros((size,size), dtype=np.float32)
rij1 = np.zeros((size,size), dtype=np.float32)
x_max = 10
x_min = -10
y_max = 10
y_min = -10

#initialization
r = np.zeros((size,2))
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

#computation
start = timer()
for i in range(size):
	for j in range(size):
		rij[i][j] = r[i] - r[j]
		rij_mag[i][j] = np.sqrt(np.square(rij[i][j][0])+np.square(rij[i][j][1]))
lifetime = timer() - start
print("Test 1 complete [linear] : " , lifetime)

#initialization
r = np.zeros((size,2))
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

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
print("Test 2 complete [linear optimzed] : " , lifetime)

#initialization
r = np.zeros((size,2),dtype=np.float32)
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

start = timer()
for i in range(size):
	for j in range(size):
		rij0[i][j] = r[i][0] - r[j][0]
		rij1[i][j] = r[i][1] - r[j][1]
for i in range(size):
	rij_mag[i] = mag_cpu(rij0[i], rij1[i])
	rij_mag = np.sqrt(rij_mag)
lifetime = timer() - start
print("Test 3 complete [vectorized] : " , lifetime)

#initialization
r = np.zeros((size,2),dtype=np.float32)
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

start = timer()
for i in range(size):
	for j in range(size):
		rij0[i][j] = r[i][0] - r[j][0]
		rij1[i][j] = r[i][1] - r[j][1]
for i in range(size):
	rij_mag[i] = mag_gpu(rij0[i], rij1[i])
	rij_mag = np.sqrt(rij_mag)
lifetime = timer() - start
print("Test 4 complete [gpu] : " , lifetime)

print("Set 11: size = 5000")

#declarations
size = 5000
r = np.zeros((size,2))
rij = np.zeros(((size,size,2)))
rij_mag = np.zeros((size,size))
rij0 = np.zeros((size,size), dtype=np.float32)
rij1 = np.zeros((size,size), dtype=np.float32)
x_max = 10
x_min = -10
y_max = 10
y_min = -10

#initialization
r = np.zeros((size,2))
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

#computation
start = timer()
for i in range(size):
	for j in range(size):
		rij[i][j] = r[i] - r[j]
		rij_mag[i][j] = np.sqrt(np.square(rij[i][j][0])+np.square(rij[i][j][1]))
lifetime = timer() - start
print("Test 1 complete [linear] : " , lifetime)

#initialization
r = np.zeros((size,2))
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

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
print("Test 2 complete [linear optimzed] : " , lifetime)

#initialization
r = np.zeros((size,2),dtype=np.float32)
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

start = timer()
for i in range(size):
	for j in range(size):
		rij0[i][j] = r[i][0] - r[j][0]
		rij1[i][j] = r[i][1] - r[j][1]
for i in range(size):
	rij_mag[i] = mag_cpu(rij0[i], rij1[i])
	rij_mag = np.sqrt(rij_mag)
lifetime = timer() - start
print("Test 3 complete [vectorized] : " , lifetime)

#initialization
r = np.zeros((size,2),dtype=np.float32)
for i in range(size):
	r[i,0] = np.random.rand() * (x_max-x_min)/2 - (x_max-x_min)/4
	r[i,1] = np.random.rand() * (y_max-y_min)/2 - (y_max-y_min)/4

start = timer()
for i in range(size):
	for j in range(size):
		rij0[i][j] = r[i][0] - r[j][0]
		rij1[i][j] = r[i][1] - r[j][1]
for i in range(size):
	rij_mag[i] = mag_gpu(rij0[i], rij1[i])
	rij_mag = np.sqrt(rij_mag)
lifetime = timer() - start
print("Test 4 complete [gpu] : " , lifetime)

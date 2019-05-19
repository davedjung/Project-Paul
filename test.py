from multiprocessing import Pool
import numpy
from timeit import default_timer as timer

def compute(a):
	sum = 0
	for i in range(a):
		sum += i
	return sum

if __name__ == '__main__':
	start = timer()
	pool = Pool()
	roots = pool.map(compute, range(100000))
	print(timer()-start)
	for i in range(100000):
		count = compute(i)
	print(timer()-start)
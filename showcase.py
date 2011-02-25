#!/usr/bin/python

def main():
	import numc as np
	a = np.random.rand(2, 3, 1, 5)
	b = np.random.rand(4, 5)
		
	print "=== NumC ==="
	test(np, a, b)
	
	import numpy as np
	print("\n=== NumPy ===")
	test(np, a, b)


def test(np, a, b):
	x = np.sum(np.sin(a) + np.square(b))
	print x
	print np.mean(np.square(b))
	
if(__name__ == "__main__"):
	main()

#EOF
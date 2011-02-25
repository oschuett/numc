#!/usr/bin/python

def main():
	import numc as np
	a = np.random.rand(3, 4, 5)
	b = np.random.rand(5)
	
	c = np.sum(np.sin(a) + np.square(b))
	print "NumC: %s"%c
	
	import numpy as np
	c = np.sum(np.sin(a) + np.square(b))
	print "NumPy: %s"%c
	
if(__name__ == "__main__"):
	main()

#EOF
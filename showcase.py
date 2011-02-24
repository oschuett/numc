#!/usr/bin/python

def main():
	import numc as np
	a = np.ones((5))
	b = np.ones((3,5))
	c = np.sum(np.sin(a) + np.square(b))
	print "NumC: %s"%c
	
	import numpy as np
	a = np.ones((5))
	b = np.ones((3,5))
	c = np.sum(np.sin(a) + np.square(b))
	print "NumPy: %s"%c
	
	
if(__name__ == "__main__"):
	main()

#EOF
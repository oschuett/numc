#!/usr/bin/python

def main():
	import numpy as np
	a = np.ones((5))
	b = np.ones((3,5))
	print np.sum(np.sin(a) + np.square(b))
	
	import numc as np
	a = np.ones((5))
	b = np.ones((3,5))
	print np.sum(np.sin(a) + np.square(b))

	
if(__name__ == "__main__"):
	main()

#EOF
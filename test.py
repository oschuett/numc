#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
import numc

#numc.set_debug(2)

def main():	
	for (k,v) in globals().items():
		if(not k.startswith("test_")): continue
		print("================= Test: %s ==================="%k)
		results1 = v(numpy)
		results2 = v(numc)
		#for r in results2:
		#	print r
		#	r.evaluate()
		failed = False
		for (r1, r2) in zip(results1, results2):
			if(r1.shape != r2.shape): failed=True
			if(r1.dtype != r2.dtype): failed=True
 				
			if(not failed and r1.size > 0):
				diff = numpy.max(numpy.abs(r1-r2))
				if(diff > 1e-15): failed=True
					
			if(failed):
				print("Numpy-Result-Shape: %s"%str(r1.shape))
				print("Numc-Result-Shape: %s"%str(r2.shape))
				print("Numpy-Result: %s"%r1)
				print("Numc-Result: %s"%r2)
				r2.__array_interface__
				print("Numc-Result(evaluated): %s"%r2)
				print("Diff: %e"%diff)
				raise(Exception("%s failed"%k))
		if(not failed):
			print("passed :-)")

def test_sliece(np):
	a = np.reshape(np.arange(1000), (10,10,10))
	return(a[1], a[-1:2:3,None,5], a[-1:2:3,np.newaxis,5] , a[-4], a[...,1], a[...,1,...,None,2], )

def test0_view(np):
	x = np.arange(5)
	y = x[::2] #creates only a view
	x[0] = 3 # changing x changes y as well, since y is a view on x
	return(x, y)
	
def test_modifiy(np):
	a = np.sin(np.zeros(5))
	b = np.square(a)
	a[2] = 123
	return (a, np.sum(b))

def test_modifiy2(np):
	a = np.zeros(5)
	b = np.sin(a)
	a[2] = 123
	return (a, np.sum(b))

def test0_no_unnecessary_copies(np):
	a = np.sin(np.zeros(5))
	b = np.square(a)
	a[2] = 123
	c = np.sin(a)
	a[3] = 42
	return (a, np.sum(b), c)

def test_add_reduce(np):
	a = np.reshape(np.arange(40), (2,4,5))
	return( np.add.reduce(np.sin(a)), )

def test_sum(np):
	a = np.reshape(np.arange(40), (2,4,5))
	return( np.sum(a, axis=1), )

def test_sum2(np):
	a = np.reshape(np.arange(1800), (2,4,5,3,5,3))
	#a = np.reshape(np.arange(40), (2,4,5))
	return( np.sum(a), )

def test_mean(np):
	#a = np.reshape(np.arange(1800), (2,4,5,3,5,3))
	a = np.reshape(np.arange(32), (2,2,2,2,2))
	return( np.mean(np.sin(a)), )

def test0_mean_with_casting(np):
	#a = np.reshape(np.arange(1800), (2,4,5,3,5,3))
	a = np.reshape(np.arange(40), (2,4,5))
	return( np.mean(a), )

if(__name__ == "__main__"):
	main()

#EOF
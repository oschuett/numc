#!/usr/bin/python
# -*- coding: utf-8 -*-

import numc as np


def run(start_matrix):
	u = start_matrix
	dx = 1.0/(nx-1)
	dy = 1.0/(ny-1)
	dx2, dy2 = dx**2, dy**2
	dnr_inv = 0.5/(dx2 + dy2)

	for i in range(5):
		u[1:-1, 1:-1] = ((u[0:-2, 1:-1] + u[2:, 1:-1])*dy2 +
      	         (u[1:-1,0:-2] + u[1:-1, 2:])*dx2)*dnr_inv



np.set_debug(0)

# Der gleiche C-Code braucht 10-mal länger nur wegen anderer Start-Werte ???

print("Fast")
nx = 500
ny = 5000
run(np.ones((nx, ny), "d"))

print("Slow")
run(np.load("start_matrix.npy"))

#EOF


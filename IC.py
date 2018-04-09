# a module for initial condition

import numpy as np
import scipy.fftpack as sp
from scipy.special import erf
from scipy.optimize import brentq

def normalize(psi, dx):
	return psi/float(np.sqrt(np.sum((np.abs(psi)**2))*dx))


# makes gaussian blob
def gauss_r(x, m, s, dx):
	arg = -.5*((x - m)/float(s))**2 + 0j
	rval = np.exp(arg)
	rval /= np.sqrt(np.sum((np.abs(rval)**2))*dx)
	return rval

# make gaussian blob but in k space
def gauss_k(kx, m, s, dk, dx):
	rval = gauss_r(kx, m, s, dk)
	rval = sp.ifft(rval)
	rval /= np.sqrt(np.sum((np.abs(rval)**2))*dx)
	return rval

# make sin wave with given k
def sin_r(x, k, dx):
	rval = np.sin(x*k) + 0j
	rval /= np.sqrt(np.sum((np.abs(rval)**2))*dx)
	return rval

def plane_wave(x, k, dx):
	rval = np.exp(k*x*1j)
	rval /= np.sqrt(np.sum((np.abs(rval)**2))*dx)
	return rval

# returns an initial phase space distribution
# axis 0 is position, axis 1 is velocity
def v_gauss(x, m, s, dx, du):
	arg = -.5*((x - m)/float(s))**2 + 0j
	r = normalize(np.exp(arg),dx) # position space gaussian
	u = normalize( np.fft.fftshift( np.abs(sp.fft(r)) ), du) # velocity space gaussian
	U, R = np.meshgrid(u, r)
	f = U*R
	f /= (f.sum())*dx*du 
	return f


# given mean, standard deviation, left edge, right edge, number of points
# erf(ri/s) > 1 - 1/N  (or there will be point crossing in our line)
def cl_gauss(m, s, le, ri, N):
	r = np.zeros(N) # eventual lagrangian position coordinate vector
	cp = (0. + np.arange(N + 2))/(N + 2) # correct proportion that should exist in each cell
	mass = lambda z, cp_: (1 + erf(z))/2. - cp_
	for i in range(1, len(cp)-1):
		i_ = i - 1
		a, b = (le - m)/s, (ri - m)/s
		r[i_] = s*brentq(mass, a, b, args=cp[i]) + m
	return r





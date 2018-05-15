# -*- coding: utf-8 -*-
"""
Created on 12.02.2015

@author: Vanessa Boehm

Helper Functions for power- and bispectra computations
"""
from __future__ import division
import numpy as np
from scipy.interpolate import splrep, splev, UnivariateSpline
from scipy.signal import argrelextrema

import matplotlib.pyplot as pl

def get_F2_kernel(k1,k2,cos):
	""" returns the F2 Kernel, see e.g. astro-ph/0112551 eq(45)
	* k1, k2:   abolute values of vectors
	* cos:      cosine of their enclosed angle
	"""
	a=17./21. #growth
	b=0.5*(k1/k2+k2/k1)*cos #shift
	c=2./21.*(3.*cos**2.-1.) #tidal

	F2=a+b+c

	return F2


def coslaw_ang(k1,k2,k3):
	""" returns the cosine of the angle enclosed k1 and k2  (negative! angular part of vector product)
	k1,k2,k3:   sides of a triangle
	"""
	cos_g=(k1**2.+k2**2.-k3**2.)/(2.*k1*k2)
	if (cos_g>1. or cos_g<-1.):
		print cos_g, k1, k2, k3
	return cos_g



def coslaw_side(k1,k2,cos):
	""" returns the third side of a triangle given two sides and an angle
	* k1,k2:    sides
	* cos:      cosine of angle enclosed by k1 and k2 (positive)
	"""
	#|L-l|=np.sqrt(L**2+l**2-2Ll)
	k3=np.sqrt(k1**2.+k2**2.-(2.*k1*k2*cos))

	return k3

def get_side2(k1,k2,ang):
	""" returns the third side of a triangle given two sides and an angle
	* k1,k2:    sides
	* cos:      cosine of angle enclosed by k1 and k2 (positive)
	"""

	k3=np.sqrt(k1**2.+k2**2.+(2.*k1*k2*np.cos(ang)))

	return k3

def get_ang23(l1,l2,l3,ang):
	""" returns the third side of a triangle given two sides and an angle
	* k1,k2:    sides
	* cos:      cosine of angle enclosed by k1 and k2 (positive)
	"""

	cosang=(-l1*ang-l3)/l2

	return cosang

def get_ang12(l1,l2,l3,ang):
	""" returns the third side of a triangle given two sides and an angle
	* k1,k2:    sides
	* cos:      cosine of angle enclosed by k1 and k2 (positive)
	"""

	cosang=(l1+l3*ang)/l2

	return cosang

def get_derivative(x, F_x, method, order=4, smooth=False):
	"""returns the derivative of F_x calculated with finite differencing
	* x : array, varying parameter
	* F : array, function of x
	* method: the method used to get the first derivative, one of ['cd','si']
	* order: order of the spline if spline interpolation is used
	"""

	methods=['cd','si','spl']
	try:
		assert(method in methods)
	except:
		raise ValueError('Invalid differentiation method!')
	if method=="spl":
		F_x=splrep(x,F_x)
		F_x=splev(x,F_x,der=1)
		w=np.ones(F_x.size)
		w[np.exp(x)<5e-3]=100
		w[np.exp(x)>1]=10
		nksp =  UnivariateSpline(x, F_x, s=10, w =w)
		pl.plot(x,nksp(x))
		pl.plot(x,F_x)
		#pl.xlim([1e-4,100])
		pl.show()
		deriv = nksp

	return deriv

# -*- coding: utf-8 -*-
"""
Created on 12.02.2015

@author: Vanessa Boehm

Helper Functions for power- and bispectra computations
"""
from __future__ import division
import numpy as np
from scipy.interpolate import splrep, splev
from scipy.signal import argrelextrema
#from scipy.interpolate import sproot
from scipy.interpolate import UnivariateSpline as US
#import pylab as pl

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
pl.ioff()
    
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
				
def get_derivative(x, F_x, method, order=4, smooth=True):
	"""returns the derivative of F_x calculated with finite differencing
	* x : array, varying parameter 
	* F : array, function of x
	* method: the method used to get the first derivative, one of ['cd','si']
	* order: order of the spline if spline interpolation is used
	"""
	
	methods=['cd','si']
	try:
		assert(method in methods)
	except:
		raise ValueError('Invalid differentiation method!')
	if smooth and method=="si":
		order=5
		print "smooth=True -> enforcing order of spline k=5"
	
	if method=="cd":
		print "derivative method: finite differencing - central difference"
		x=np.asarray(x)
		F_x=np.asarray(F_x)
		f_x=np.empty(len(F_x))
		f_x[1:-1]=(F_x[2::]-F_x[:-2:])/(x[2::]-x[:-2:])
		f_x[0]=(F_x[1]-F_x[0])/(x[1]-x[0])
		f_x[-1]=(F_x[-2]-F_x[-1])/(x[-2]-x[-1])
		
	if method=="si":
		print "derivative method: differentiating interpolation spline"
		#get spline representation of input function
  
		F_i=splrep(x,F_x,k=order,quiet=1)
##test plot 1
		pl.figure()
		pl.plot(x,F_x)
		x_=np.linspace(min(x),max(x),200)
		pl.plot(x_,splev(x_,F_i))
		pl.savefig('Test1.png')

		f_x=splev(x,F_i,der=1,ext=2)
	
	if smooth:
		max_index=argrelextrema(f_x,np.greater)		
		min_index=argrelextrema(f_x,np.less)

		mean_x=[]
		mean_f=[]
		for i in xrange(0,len(min_index[0])-1):

#			mean=(f_x[max_index[0][i]]+f_x[min_index[0][i]])/2.
#			idx = np.argmin(np.abs(f_x[np.arange(min_index[0][i],max_index[0][i])] - mean))
#			mean_x+=[x[idx+min_index[0][i]]]
#			mean_f+=[mean]
			mean=(f_x[max_index[0][i]]+f_x[min_index[0][i+1]])/2.
			idx = np.argmin(np.abs(f_x[np.arange(max_index[0][i],min_index[0][i+1])] - mean))
			mean_x+=[x[idx+max_index[0][i]]]
			mean_f+=[mean]

		mean_x=np.asarray(mean_x)
		mean_f=np.asarray(mean_f)
		
		while min(mean_x)<-4.5:
			mean_x=mean_x[1::]
			mean_f=mean_f[1::]
		while max(mean_x)>-0.1:
			mean_x=mean_x[0:-1:]
			mean_f=mean_f[0:-1:]
		print min(mean_x),max(mean_x)

		x_new=np.concatenate((x[np.where(x<-4.5)],mean_x,x[np.where(x>-0.1)]))  

##test plot 2
		pl.figure()
		pl.semilogx(np.exp(x),f_x)
		f_i1=splrep(x,f_x,k=3,quiet=1,s=0.0)
		x_=np.linspace(min(x),max(x),400)
		pl.plot(np.exp(x_),splev(x_,f_i1),ls="--")
		pl.plot(np.exp(mean_x),mean_f,"ro")
		pl.savefig('Test2.png')
		
		f_new=np.concatenate((f_x[np.where(x<-4.5)],mean_f,f_x[np.where(x>-0.1)]))

		deriv=splrep(x_new,f_new,k=3,quiet=1)
## test plot 3
		pl.figure()
		pl.semilogx(np.exp(x)/0.7,f_x)
		pl.plot(np.exp(x_new)/0.7,f_new,ls="--",c="g")
		pl.plot(np.exp(x_)/0.7,splev(x_,deriv),ls=":",c="r")
#		pl.xlim(0.01,0.5)
#		pl.ylim(-3.,0.5)
		pl.savefig('Test3.pdf')

	return deriv

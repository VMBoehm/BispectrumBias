# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:47:19 2016

@author: VBoehm
"""
from __future__ import division

cimport cython
import numpy as np
from libc.math cimport sin
cimport numpy as np
from libc.stdlib cimport malloc, free
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPEF = np.float
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float_t DTYPEF_t

DTYPEI = np.int
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPEI_t

#cdef double sin_func(double x):
#    return sin(x)
				
cdef double my_sum(double[:] input_array):
    cdef int i
    cdef int dim1=len(input_array)
    cdef double result=0
    for i from 1 <= i < dim1-1:			
        result+=input_array[i]
    return result					
					
def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)

def _basic_simps(y, start, stop, x, dx, axis):
    nd = len(y.shape)
    if start is None:
        start = 0
    step = 2
    slice_all = (slice(None),)*nd
    slice0 = tupleset(slice_all, axis, slice(start, stop, step))
    slice1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
    slice2 = tupleset(slice_all, axis, slice(start+2, stop+2, step))

    if x is None:  # Even spaced Simpson's rule.
        result = np.sum(dx/3.0 * (y[slice0]+4*y[slice1]+y[slice2]),
                        axis=axis)
    else:
        # Account for possibly different spacings.
        #    Simpson's rule changes a bit.
        h = np.diff(x, axis=axis)
        sl0 = tupleset(slice_all, axis, slice(start, stop, step))
        sl1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
        h0 = h[sl0]
        h1 = h[sl1]
        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = h0 / h1
        tmp = hsum/6.0 * (y[slice0]*(2-1.0/h0divh1) +
                          y[slice1]*hsum*hsum/hprod +
                          y[slice2]*(2-h0divh1))
        result = np.sum(tmp, axis=axis)
    return result

def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)
    
def simps(y, x=None, dx=1, axis=-1, even='avg'):
    """
    Integrate y(x) using samples along the given axis and the composite
    Simpson's rule.  If x is None, spacing of dx is assumed.
    If there are an even number of samples, N, then there are an odd
    number of intervals (N-1), but Simpson's rule requires an even number
    of intervals.  The parameter 'even' controls how this is handled.
    Parameters
    ----------
    y : array_like
        Array to be integrated.
    x : array_like, optional
        If given, the points at which `y` is sampled.
    dx : int, optional
        Spacing of integration points along axis of `y`. Only used when
        `x` is None. Default is 1.
    axis : int, optional
        Axis along which to integrate. Default is the last axis.
    even : {'avg', 'first', 'str'}, optional
        'avg' : Average two results:1) use the first N-2 intervals with
                  a trapezoidal rule on the last interval and 2) use the last
                  N-2 intervals with a trapezoidal rule on the first interval.
        'first' : Use Simpson's rule for the first N-2 intervals with
                a trapezoidal rule on the last interval.
        'last' : Use Simpson's rule for the last N-2 intervals with a
               trapezoidal rule on the first interval.
    Notes
    -----
    For an odd number of samples that are equally spaced the result is
    exact if the function is a polynomial of order 3 or less.  If
    the samples are not equally spaced, then the result is exact only
    if the function is a polynomial of order 2 or less.
    """
    y = np.asarray(y)
    nd = len(y.shape)
    N = y.shape[axis]
    last_dx = dx
    first_dx = dx
    returnshape = 0
    if x is not None:
        x = np.asarray(x)
        if len(x.shape) == 1:
            shapex = [1] * nd
            shapex[axis] = x.shape[0]
            saveshape = x.shape
            returnshape = 1
            x = x.reshape(tuple(shapex))
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-d or the "
                             "same as y.")
        if x.shape[axis] != N:
            raise ValueError("If given, length of x along axis must be the "
                             "same as y.")
    if N % 2 == 0:
        val = 0.0
        result = 0.0
        slice1 = (slice(None),)*nd
        slice2 = (slice(None),)*nd
        if even not in ['avg', 'last', 'first']:
            raise ValueError("Parameter 'even' must be "
                             "'avg', 'last', or 'first'.")
        # Compute using Simpson's rule on first intervals
        if even in ['avg', 'first']:
            slice1 = tupleset(slice1, axis, -1)
            slice2 = tupleset(slice2, axis, -2)
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val += 0.5*last_dx*(y[slice1]+y[slice2])
            result = _basic_simps(y, 0, N-3, x, dx, axis)
        # Compute using Simpson's rule on last set of intervals
        if even in ['avg', 'last']:
            slice1 = tupleset(slice1, axis, 0)
            slice2 = tupleset(slice2, axis, 1)
            if x is not None:
                first_dx = x[tuple(slice2)] - x[tuple(slice1)]
            val += 0.5*first_dx*(y[slice2]+y[slice1])
            result += _basic_simps(y, 1, N-2, x, dx, axis)
        if even == 'avg':
            val /= 2.0
            result /= 2.0
        result = result + val
    else:
        result = _basic_simps(y, 0, N-2, x, dx, axis)
    if returnshape:
        x = x.reshape(saveshape)
    return result
    
#from cython_gsl cimport *
#from cython_gsl cimport gsl_interp
#
#def interp1d_gsl (np.ndarray[DTYPEF_t, ndim=1] x_new, np.ndarray[DTYPEF_t, ndim=1] x_, np.ndarray[DTYPEF_t, ndim=1] y_):
#    assert(len(x_)==len(y_))
#    cdef unsigned int i
#    cdef unsigned int n =len(x_)
#    cdef unsigned int m =len(x_new)
#    cdef np.ndarray[DTYPEF_t, ndim=1] y_new = np.zeros(m,dtype=x_new.dtype)
#    
#    cdef double *x
#    cdef double *y
#
#    x = <double *>malloc(n*cython.sizeof(double))
#    if x is NULL:
#      raise MemoryError()
#    
#    y = <double *>malloc(n*cython.sizeof(double))
#    if y is NULL:
#      raise MemoryError()
#
#    for i in xrange(n):
#      x[i] = x_[i]
#      y[i] = y_[i]
#
#    cdef gsl_interp_accel *acc
#    acc = gsl_interp_accel_alloc ()
#    cdef gsl_spline *spline
#    spline = gsl_spline_alloc (gsl_interp_linear, n)
#
#    gsl_spline_init (spline, x, y, n)
#
#    for i  from 0 <= i < m:
#        y_new[i] = gsl_spline_eval (spline, x_new[i], acc)
#
#    gsl_spline_free (spline)
#    gsl_interp_accel_free (acc)
#    
#    return y_new
#def get_cl2(np.ndarray[DTYPEF_t, ndim=2] cl_unlen, np.ndarray[DTYPEF_t, ndim=2] cl_tot, np.ndarray[DTYPEF_t, ndim=2] l2, int lmin, int lmax):
#	
#	## If l2 lies in range of known Cl's, linearly interpolate to get cl2	
#   cdef np.ndarray[DTYPEF_t, ndim=2] cl2_unlen = np.zeros(np.shape(l2), dtype=cl_unlen.dtype)
#   cdef np.ndarray[DTYPEF_t, ndim=2] cdcl2_tot = np.zeros(np.shape(l2), dtype=cl_unlen.dtype)
#   cdef np.ndarray[DTYPEF_t, ndim=2] deltal    = np.zeros(np.shape(l2), dtype=cl_unlen.dtype)
#   x,y = l2.shape
#   index=np.arange(y)
#   for i in range(x):
#     l2i =l2[i]
#     idxs1 = index[np.where( np.logical_and( lmin < l2i, l2i < lmax))]
#     idxs2 = index[np.where( l2i <= lmin )]
#     idxs3 = index[np.where( l2i >= lmax )]
#   
#     lowl = np.floor(l2).astype(int)
#     highl = np.ceil(l2).astype(int)
#     deltal[idxs1] = l2[idxs1] - lowl[idxs1]
#   deltal[idxs2] = lmin - l2[idxs2]
#   deltal[idxs3] = l2[idxs3] - lmax
#   
#   lowl -= lmin
#   highl -= lmin
#   
#   cl2_tot[idxs1] = cl_tot[lowl[idxs1]] + deltal[idxs1] * (cl_tot[highl[idxs1]] - cl_tot[lowl[idxs1]])
#   cl2_unlen[idxs1] = cl_unlen[lowl[idxs1]] + deltal[idxs1] * (cl_unlen[highl[idxs1]] - cl_unlen[lowl[idxs1]]) 
#	
#   cl2_tot[idxs2] = cl_tot[0] + deltal[idxs2] * (cl_tot[0] - cl_tot[1]) 
#   cl2_unlen[idxs2] = cl_unlen[0] + deltal[idxs2] * (cl_unlen[0] - cl_unlen[1])
#	
#   cl2_tot[idxs3] = cl_tot[lmax-lmin] + deltal[idxs3]*(cl_tot[lmax-lmin] - cl_tot[lmax-lmin-1]) * np.exp(-deltal[idxs3]**2) 
#   cl2_unlen[idxs3] = cl_unlen[lmax-lmin] + deltal[idxs3]*(cl_unlen[lmax-lmin] - cl_unlen[lmax-lmin-1]) * np.exp(-deltal[idxs3]**2)
#	
#   return cl2_unlen, cl2_tot
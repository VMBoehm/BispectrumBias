# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 13:53:29 2015

@author: Vanessa M. Boehm
"""


"""Useful Constants"""
### everything is in Megaparsec
AU          = 1.496*1e8 #km
PC          = AU/(4.8481*1e-6) #km
KM2MPC      = 1./PC/1e6
MPC2KM      = PC*1e6
HUBBLE_UNIT = 100.* KM2MPC #in 1/sec
LIGHT_SPEED = 299792458.*1e-3 #km/s
SEC2YEAR    = 1./(60.*60.*24.*365.2425)
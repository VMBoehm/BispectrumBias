
# coding: utf-8

# In[173]:

get_ipython().magic(u'matplotlib inline')
import csv
import pickle
import numpy as np
import matplotlib.pyplot as pl
from scipy.integrate import simps
from scipy.interpolate import interp1d
from collections import namedtuple
import itertools
import datetime as dt
import scipy.stats as stats
from scipy.interpolate import interp1d
import scipy.signal as signal
matplotlib.rcParams.update({'font.size': 16})

# In[174]:

filename='dndz_LSST_i27_SN5_3y'
f = open(filename+".csv_manually_extrapolated")
csv_f = csv.reader(f)


# In[175]:

z =[]
dn=[]
for row in csv_f:
    z+=[float(row[0])]
    dn+=[float(row[1])]
z = np.asarray(z)
dn= np.asarray(dn)
dn[0]=0


# In[176]:

pl.figure()
pl.semilogx(z,dn)
pl.show()


                
                
# In[177]:

interp_dn=interp1d(z, dn, kind='linear')
norm=simps(interp_dn(z),z)
norm


# In[178]:

pickle.dump([z,dn],open(filename+'tot_extrapolated.pkl','w'))


bin0=[0,0.5,0.03]
bin1=[0.5,1,0.03]
bin2=[1.5,0.5,0.04]
bin3=[2.,1.5,0.05]
bin4=[3.5,0.5,0.05]

def gauss(x,errscale):
    var =(errscale*(1+x))**2
    res = np.exp(-x**2/2/var)
    norm= simps(res,x)
    return res/norm

delta = 1e-4
big_grid = np.arange(-6,6,delta)


# In[188]:




# In[248]:

c=['b','c','g','orange','r']
a=0.8
ii=0
z_=np.linspace(0,6,200)
fig = pl.figure(figsize = (6,4))
pl.plot(z,dn,color='k',lw=2)
ax = fig.add_subplot(111)
z_ = np.linspace(0,6,100)
res=[]
for mbin in [bin0,bin1,bin2,bin3,bin4]:
    z_      = np.linspace(mbin[0],mbin[0]+mbin[1],100)
    dndz    = interp_dn(z_)
    dndz    = interp1d(z_, dndz, kind='linear',bounds_error=False,fill_value=0.)
    norm    = simps(dndz(big_grid),big_grid)
    errscale= mbin[2]
    pmf1 = dndz(big_grid)*delta
    pmf2 = gauss(big_grid,errscale)*delta
    conv_pmf = signal.fftconvolve(pmf1,pmf2,'same')/delta
    print norm/simps(conv_pmf,big_grid)
    val     = interp1d(big_grid,conv_pmf,kind='linear',bounds_error=False,fill_value=0.)
    ax.fill_between(z_,0,val(z_),color=c[ii],alpha=a,lw=0)
    a-=0.08
    ii+=1
    res+=[mbin,conv_pmf,norm]
ax.set_xlim(0,4.2)
ax.set_ylim(-1,60)
ax.set_xlabel('$z$',fontsize=20)
ax.set_ylabel(r'$\mathrm{d}N/\mathrm{d}z\ [\mathrm{arcmin}^{-2}]$',fontsize=20)
pl.savefig('RedshiftBins.pdf',bbox_inches='tight')


pickle.dump([big_grid,res],open(filename+'_extrapolated.pkl','w'))

# In[172]:

#for mbin in [bin0,bin1,bin2]:
#    simple = stats.uniform(loc=mbin[0],scale=bin0[1])
#
#    pmf1 = simple.pdf(big_grid)*delta
#    pmf2 = gauss(big_grid,errscale)*delta
#    conv_pmf = signal.fftconvolve(pmf1,pmf2,'same') # Convolved probability mass function
#
#    conv_pmf = conv_pmf/sum(conv_pmf)
#
#    pl.plot(big_grid,pmf1/delta, label='Tophat')
#    #pl.plot(big_grid,pmf2/delta, label='Gaussian error')
#    pl.plot(big_grid,conv_pmf/delta, label='Sum')
#    pl.xlim(-1,max(big_grid))
#    #pl.legend(loc='best'), pl.suptitle('PMFs')
#    pl.ylim(-0.5,3)
#    
#    print simps(conv_pmf/delta,big_grid)
#    print simps(pmf1/delta,big_grid)
#    print simps(pmf2/delta,big_grid)
#    
#
#
## In[164]:
#
#
#
#
## In[166]:
#
#
#
#
## In[167]:
#
##pl.plot(big_grid,pmf1/delta, label='Tophat')
##pl.plot(big_grid,pmf2/delta, label='Gaussian error')
#
#
#
## In[ ]:
#
#
#

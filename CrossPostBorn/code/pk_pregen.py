#!usr/bin/env python
#
# Some code primarily for playing around for now.
# Makes a class which combines the pre-computed P(k) tables
# to return a single P(k) given the parameters.
#
from __future__ import print_function,division

import numpy as np
import tools
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate


def cleft(lpt,b1,b2,bs,bn,alpha,sn,auto, ZVfiles=False):
    """  
    Returns P(k) given the parameters for the array
    """
    # The column order is:
    # k,P_Z,P_A,P_W,P_d,P_{dd},P_{d^2},
    # P_{d^2d^2},P_{dd^2},P_{s^2},P_{ds^2},
    # P_{d^2s^2},P_{s^2s^2},P_{D2d},P_{dD2d}
    if auto:
        par = np.array([0,1.,1.,1.,b1,b1*b1,b2,b2*b2,b1*b2,\
                            bs,b1*bs,b2*bs,bs*bs,bn,b1*bn])
        if ZVfiles: par[7] /= 4. #Bad normalization in ZV's files
    else:
        par = np.array([0,1.,1.,1.,b1/2.,0,b2/2.,0,0,bs/2.,0,0,0,bn/2.,0])
    tmp = np.dot(lpt,par)
    tmp-= alpha/2.*lpt[:,0]**2*lpt[:,1]
    return( (lpt[:,0],tmp+sn) )

def cleft_der(pktable,b1,b2,bs,bn,alpha,sn, pnum, auto):
    """  
    Returns the derivative of P(k) from pktable 
    w.r.t. parameter number pnum, with the "usual" order 
    0: b1, 1:b2, 2:bs, 3:bn, 4:alpha, 5:sn, at the 
    position in parameter space passed at "p"

    """
    # The column order is:
    # k,P_Z,P_A,P_W,P_d,P_{dd},P_{d^2},
    # P_{d^2d^2},P_{dd^2},P_{s^2},P_{ds^2},
    # P_{d^2s^2},P_{s^2s^2},P_{D2d},P_{dD2d}
    if auto:
        if pnum==0:     # b1
            dP = pktable[:,4] + 2*b1*pktable[:, 5] \
                 + b2*pktable[:, 8] + bs*pktable[:, 10] + bn*pktable[:, 14]
        elif pnum==1:	# b2
            dP = pktable[:,6] + 2*b2*pktable[:, 7] / 4. 
            + b1*pktable[:,8] + bs*pktable[:, 11]
        elif pnum==2:	# bs
            dP = pktable[:,9] + 2*bs*pktable[:, 12] \
                 + b1*pktable[:, 10]  + b2*pktable[:, 11] 
        elif pnum==3:	# bn
            dP = pktable[:,13] + b1*pktable[:, 14] 
        elif pnum==4:	# alpha
            dP =-0.5*pktable[:,0]**2*pktable[:,1]
        elif pnum==5:	# sn
            dP = 1.0*np.ones_like(pktable[:,0])
        else:
            raise RuntimeError("Unknown pnum="+str(pnum))

    else:
        if pnum==0:		# b1
            dP = 0.5*pktable[:,4]
        elif pnum==1:	# b2
            dP = 0.5*pktable[:,6]
        elif pnum==2:	# bs
            dP = 0.5*pktable[:,9]
        elif pnum==3:	# bn
            dP = 0.5*pktable[:,13]
        elif pnum==4:	# alpha
            dP =-0.5*pktable[:,0]**2*pktable[:,1]
        elif pnum==5:	# sn
            dP = 1.0*np.ones_like(pktable[:,0])
        else:
            raise RuntimeError("Unknown pnum="+str(pnum))
    return( (pktable[:,0],dP) )


class PkCLEFT:
    """   
    A class which holds the pre-computed P(k) table and has a single
    method which returns the prediction gives the bias parameters.
    """
#
    def __init__(self,pktable=None, pktabfile=None, db = None, z = None):
        """    
        Initializes the class by reading in the pre-computed P(k) table.
        The power spectrum table filename is passed as a parameter, if
        nothing is passed a DarkSky cosmology at z=1 is used.
        """
        # Read the data from pre-computed file.
        if pktable is None:
            if pktabfile==None:            
                print('Either need a table or the path of the file to read the table from')
                
            else: self.pktable=np.loadtxt(pktabfile)
        else:
            self.pktable=pktable

        #
    def __call__(self,b1,b2,bs,bn,alpha,sn,auto=False):
        """  
        Returns P(k) given the parameters.
        """
        return cleft(self.pktable,b1,b2,bs,bn,alpha,sn,auto)


    def deriv(self, b1,b2,bs,bn,alpha,sn, pnum, auto=False):
        """    
        Returns the derivative of P(k) w.r.t. parameter number pnum, with
        the "usual" order 0: b1, 1:b2, 2:bs, 3:bn, 4:alpha, 5:sn, at the 
        position in parameter space passed at "p"
        """
        return cleft_der(self.pktable, b1, b2, bs, bn, alpha, sn, pnum, auto)



    
class PkZCLEFT():
    #### DOES NOT SUPPORT bn ####
    '''Given a pat h (db), redshift(z), varioation(vary) and CLEFT fit params(p)
    interpolate over sigma8 assuming the fodler tree-
    db/ps_response/response_files
    where the template of file names is -
    basefile = "db/ps00_hh_RunPB_46_z%03d.dat"%iz
    responsefile = "db/ps_response/ps00_hh_RunPB_46_z%03d_rp+%02d.dat"%(iz, vary)
    '''
    

    def __init__(self, zmin, zmax, z0=None, db="../../data/data-generated/pkcleft/"):
        '''
        '''
        self.db = db
        self.zmin, self.zmax = zmin, zmax
        self.iz = np.arange(np.round(zmin, 1), np.round(zmax, 1), 0.1)
        self.fn = self.db +"pkcleft_zz%03d.dat"

        self.rsp = self._readfiles()
        self.intpf = self._interp()

        self.z0 = z0
        if self.z0 is not None:
            print('Fixed at z = %0.2f'%self.z0)
            wts = tools.lagrange2(self.z0, self.iz)
            self.rspz0 = (wts.reshape(-1, 1, 1)*self.rsp).sum(axis = 0)
        
        #self.pardict = {'zz':-1, 'b1':0, 'b2':1, 'bs2':2, 'bn':3, 'alpha':4, 'sn':5}

    def __call__(self, par ,auto=False):
        """  
        Returns P(k) given the parameters.
        Order in which params are attributed is zz, b1, b2, bs2, bn, alpha, sn
        If z0 is fixed, order in which params are attributed is b1, b2, bs2, bn, alpha, sn

        """
        if self.z0 is not None: par = [self.z0] + par
        if len(par) == 1:
            zz, b1, b2, bs2, bn, alpha, sn = par[0], 0, 0, 0, 0, 0, 0
        elif len(par) == 2:
            zz, b1, b2, bs2, bn, alpha, sn = par[0], par[1], 0, 0, 0, 0, 0
        elif len(par) == 3:
            zz, b1, b2, bs2, bn, alpha, sn = par[0], par[1], par[2], 0, 0, 0, 0
        elif len(par) == 4:
            zz, b1, b2, bs2, bn, alpha, sn = par[0], par[1], par[2], par[3], 0, 0, 0
        elif len(par) == 5:
            zz, b1, b2, bs2, bn, alpha, sn = par[0], par[1], par[2], par[3], par[4], 0, 0
        elif len(par) == 6:
            zz, b1, b2, bs2, bn, alpha, sn = par[0], par[1], par[2], par[3], par[4], par[5], 0
        elif len(par) == 7:
            zz, b1, b2, bs2, bn, alpha, sn = par[0], par[1], par[2], par[3], par[4], par[5], par[6]
        else:
            print('Length of par not recognized. par = ', par)

        if self.z0 is None:
            ##Interpolate with scipy interpolate
            #rsp = np.zeros_like(self.intpf, dtype = 'f8')
            #for foo in range(rsp.shape[1]):
            #    rsp[:, foo] = np.array([f(zz) for f in self.intpf[:, foo]])

            ##Interpolate with lagrange method
            wts = tools.lagrange2(zz, self.iz)
            rsp = (wts.reshape(-1, 1, 1)*self.rsp).sum(axis = 0)
        else:
            rsp = self.rspz0
        p = [b1, b2, bs2, bn, alpha, sn, auto]
        return cleft(rsp, *p)

    def pars(self, z=None, b1=0, b2=0, bs2=0, bn=0, alpha=0, sn=0):
        if z is None:
            if self.z0 is not None:
                return [b1, b2, bs2, bn, alpha, sn]
            else:
                print('\n### Need to assign redshift since it is not fixed ###\n')
                sys.exit()
        else:
            if self.z0 is not None:
                print('\n### Redshift is already assigned cannot override ###\n')
                sys.exit()
            else: return [z, b1, b2, bs2, bn, alpha, sn]

    
    def pkz(self, z):
        return PkZCLEFT(zmin=self.zmin, zmax=self.zmax, z0=z, db=self.db)
        
    
    def _readfiles(self):
        '''Read the response files'''
        files = []
        for i in self.iz:
            files.append(np.loadtxt(self.fn%(i*100)))
        return np.array(files)


    def _interp(self):
        '''Create interpolation function array to go from sigma8 to reponse array'''
        intpf = []
        rsp = self.rsp
        for foo in range(rsp.shape[1]):
            for boo in range(rsp.shape[2]):
                intpf.append(interpolate(self.iz, rsp[:, foo, boo]))
        
        intpf = (np.array(intpf)).reshape(rsp[0].shape)
        return intpf


class Pk8CLEFT(PkCLEFT):
    #### DOES NOT SUPPORT bn ####
    '''Given a pat h (db), redshift(z), varioation(vary) and CLEFT fit params(p)
    interpolate over sigma8 assuming the fodler tree-
    db/ps_response/response_files
    where the template of file names is -
    basefile = "db/ps00_hh_RunPB_46_z%03d.dat"%iz
    responsefile = "db/ps_response/ps00_hh_RunPB_46_z%03d_rp+%02d.dat"%(iz, vary)
    '''
    

    def __init__(self, db =  "../data/RunPB/",  z= 2, vary = [3, 5]):
        '''
        '''
        self.db = db
        self.dbpt = db + 'PTdata/'
        self.z = z
        self.iz = z*100
        self.fn = self.dbpt +"ps00_hh_RunPB_46_z%03d.dat"%self.iz
        self.lin = np.loadtxt(db +'class_output/RunPB00_z%03d_pk.dat'%self.iz)
        
        self.sig80 = 0.8195#tools.sig8(self.lin[:, 0], self.lin[:, 1])
        ###super().__init__(self.fn)
        self.lpt = PkCLEFT(fn = self.fn)
        self.vary = np.array(vary)
        self.vary.sort()
        self.sig8s = self._sig8s()
        self.rsp = self._readfiles()
        self.intpf = self._interp()
        self.pardict = {'s8':-1, 'b1':0, 'b2':1, 'bs2':2, 'bn':3, 'alpha':4, 'sn':5}

    def __call__(self, par ,auto=False):
        """  
        Returns P(k) given the parameters.
        """
        if len(par) == 2:
            p = [0, 0, 0, 0] + list(par) +[auto]
            return self.lpt(*p)
        elif len(par) == 5:
            p = list(par[:3]) +[0] +  list(par[3:])+ [auto]
            return self.lpt(*p)
        elif len(par) == 6:
            s8 = par[0]
            p = list(par[1:4]) + [0] + list(par[4:]) + [auto]
            #rsp = np.zeros_like(self.intpf, dtype = 'f8')
            #for foo in range(rsp.shape[1]):
            #    rsp[:, foo] = np.array([f(s8) for f in self.intpf[:, foo]])
            wts = tools.lagrange2(s8, self.sig8s)
            rsp = (wts.reshape(-1, 1, 1)*self.rsp).sum(axis = 0)
            return cleft(rsp, *p)

    def deriv(self, par, name, auto=False):
        """    
        Returns the derivative of P(k) w.r.t. parameter number pnum, with
        the "usual" order 0: s8, 1: b1, 2:b2, 3:bs, 4:bn, 5:alpha, 6:sn, at the 
        position in parameter space passed at "p"
        """
        pnum = self.pardict[name]
        if pnum < 0:
            if len(par) == 6:
                s8 = par[0]
                p = list(par[1:4]) + [0] + list(par[4:])
                x1, x2, y = s8*1.01, s8*0.99, []
                for x in [x1, x2]:
                    wts = tools.lagrange2(x, self.sig8s)
                    rsp = (wts.reshape(-1, 1, 1)*self.rsp).sum(axis = 0)
                    y.append(cleft(rsp, *p, auto = auto))
                der = (y[0][1] - y[1][1])/(x1 - x2)
                return (y[0][0], der)

            else:
                print('For sigma8, parameter length should be 6')

        if pnum >= 0:            
            if len(par) == 2:
                p = [0, 0, 0, 0] + list(par)
                return self.lpt.deriv(*p, pnum = pnum, auto = auto)
            elif len(par) == 5:
                p = list(par[:3]) +[0] +  list(par[3:])
                return self.lpt.deriv(*p, pnum = pnum, auto= auto)
            elif len(par) == 6:
                s8 = par[0]
                p = list(par[1:4]) + [0] + list(par[4:])
                wts = tools.lagrange2(s8, self.sig8s)
                rsp = (wts.reshape(-1, 1, 1)*self.rsp).sum(axis = 0)
                return cleft_der(rsp, *p, pnum = pnum, auto = auto)



    def _readfiles(self):
        '''Read the response files'''
        files = []
        db = self.dbpt + 'ps_response/'
        for i in self.vary[::-1]:
            files.append(np.loadtxt(db + "ps00_hh_RunPB_46_z%03d_rp-%02d.dat"%(self.iz, i)))
        files.append(np.loadtxt(self.fn))
        for i in self.vary:
            files.append(np.loadtxt(db + "ps00_hh_RunPB_46_z%03d_rp+%02d.dat"%(self.iz, i)))
        return np.array(files)


    def _sig8s(self):
        '''For given 'vary', create array of sigma8s'''
        sig8s = []
        for i in self.vary[::-1]:
            sig8s.append(1 - i/100.)
        sig8s.append(1)
        for i in self.vary:
            sig8s.append(1 + i/100.)
        return np.array(sig8s)*self.sig80

    def _interp(self):
        '''Create interpolation function array to go from sigma8 to reponse array'''
        intpf = []
        rsp = self.rsp
        for foo in range(rsp.shape[1]):
            for boo in range(rsp.shape[2]):
                intpf.append(interpolate(self.sig8s, rsp[:, foo, boo]))
        
        intpf = (np.array(intpf)).reshape(rsp[0].shape)
        return intpf








class PkZCLEFT():
    #### DOES NOT SUPPORT bn ####
    '''Given a pat h (db), redshift(z), varioation(vary) and CLEFT fit params(p)
    interpolate over sigma8 assuming the fodler tree-
    db/ps_response/response_files
    where the template of file names is -
    basefile = "db/ps00_hh_RunPB_46_z%03d.dat"%iz
    responsefile = "db/ps_response/ps00_hh_RunPB_46_z%03d_rp+%02d.dat"%(iz, vary)
    '''
    

    def __init__(self, zmin, zmax, z0=None, db="../../data/data-generated/pkcleft/"):
        '''
        '''
        self.db = db
        self.zmin, self.zmax = zmin, zmax
        self.iz = np.arange(np.round(zmin, 1), np.round(zmax, 1), 0.1)
        self.fn = self.db +"pkcleft_zz%03d.dat"

        self.rsp = self._readfiles()
        self.intpf = self._interp()

        self.z0 = z0
        if self.z0 is not None:
            print('Fixed at z = %0.2f'%self.z0)
            wts = tools.lagrange2(self.z0, self.iz)
            self.rspz0 = (wts.reshape(-1, 1, 1)*self.rsp).sum(axis = 0)
        
        #self.pardict = {'zz':-1, 'b1':0, 'b2':1, 'bs2':2, 'bn':3, 'alpha':4, 'sn':5}

    def __call__(self, par ,auto=False):
        """  
        Returns P(k) given the parameters.
        Order in which params are attributed is zz, b1, b2, bs2, bn, alpha, sn
        If z0 is fixed, order in which params are attributed is b1, b2, bs2, bn, alpha, sn

        """
        if self.z0 is not None: par = [self.z0] + par
        if len(par) == 1:
            zz, b1, b2, bs2, bn, alpha, sn = par[0], 0, 0, 0, 0, 0, 0
        elif len(par) == 2:
            zz, b1, b2, bs2, bn, alpha, sn = par[0], par[1], 0, 0, 0, 0, 0
        elif len(par) == 3:
            zz, b1, b2, bs2, bn, alpha, sn = par[0], par[1], par[2], 0, 0, 0, 0
        elif len(par) == 4:
            zz, b1, b2, bs2, bn, alpha, sn = par[0], par[1], par[2], par[3], 0, 0, 0
        elif len(par) == 5:
            zz, b1, b2, bs2, bn, alpha, sn = par[0], par[1], par[2], par[3], par[4], 0, 0
        elif len(par) == 6:
            zz, b1, b2, bs2, bn, alpha, sn = par[0], par[1], par[2], par[3], par[4], par[5], 0
        elif len(par) == 7:
            zz, b1, b2, bs2, bn, alpha, sn = par[0], par[1], par[2], par[3], par[4], par[5], par[6]
        else:
            print(par)

        if self.z0 is None:
            ##Interpolate with scipy interpolate
            #rsp = np.zeros_like(self.intpf, dtype = 'f8')
            #for foo in range(rsp.shape[1]):
            #    rsp[:, foo] = np.array([f(zz) for f in self.intpf[:, foo]])

            ##Interpolate with lagrange method
            wts = tools.lagrange2(zz, self.iz)
            rsp = (wts.reshape(-1, 1, 1)*self.rsp).sum(axis = 0)
        else:
            rsp = self.rspz0
        p = [b1, b2, bs2, bn, alpha, sn, auto]
        return cleft(rsp, *p)

    def pars(self, z=None, b1=0, b2=0, bs2=0, bn=0, alpha=0, sn=0):
        if z is None:
            if self.z0 is not None:
                return [b1, b2, bs2, bn, alpha, sn]
            else:
                print('\n### Need to assign redshift since it is not fixed ###\n')
                sys.exit()
        else:
            if self.z0 is not None:
                print('\n### Redshift is already assigned cannot override ###\n')
                sys.exit()
            else: return [z, b1, b2, bs2, bn, alpha, sn]

    
    def pkz(self, z):
        return PkZCLEFT(zmin=self.zmin, zmax=self.zmax, z0=z, db=self.db)
        
    
    def _readfiles(self):
        '''Read the response files'''
        files = []
        for i in self.iz:
            files.append(np.loadtxt(self.fn%(i*100)))
        return np.array(files)


    def _interp(self):
        '''Create interpolation function array to go from sigma8 to reponse array'''
        intpf = []
        rsp = self.rsp
        for foo in range(rsp.shape[1]):
            for boo in range(rsp.shape[2]):
                intpf.append(interpolate(self.iz, rsp[:, foo, boo]))
        
        intpf = (np.array(intpf)).reshape(rsp[0].shape)
        return intpf

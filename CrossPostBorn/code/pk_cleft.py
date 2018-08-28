#Generate the CLEFT power spectrum at different redshifts and save them
#at the desired path (currently: ../../data/data-generated/)

import numpy as np
import sys, os

#path_to_cleft = '/home/chirag/Research/codes/CLEFT_GSM/ps_py3_package/'
path_to_cleft = '/global/homes/c/chmodi/Programs/Py_codes/CLEFT_GSM/ps_py3_package'
sys.path.append(path_to_cleft)

import cleftpool as cpool

#pkfile = '/home/chirag/Research/Projects/qso_kappacmb/data/pklin_ds14.dat'
#pkfile = '/home/chirag/Research/Projects/BispectrumBias/CrossPostBorn/data/RunPB/class_output/RunPB00_z000_pk.dat'
pkfile = '../data/RunPB/pklin_RunPB.txt'
pklin = np.loadtxt(pkfile).T

print(pklin.shape)

saveqfile = '../data/RunPB/PTdata/q_kernel.dat'
saverfile = '../data/RunPB/PTdata/r_kernel.dat'

if os.path.isfile(saveqfile): qfile = saveqfile
else: qfile = None
if os.path.isfile(saverfile): rfile = saverfile
else: rfile = None

qfile, rfile = None, None

#Main function call
#cl = cpool.CLEFT(k=pklin[0], p=pklin[1], npool=32, qfile=qfile, rfile=rfile)
cl = cpool.CLEFT(pfile = pkfile, npool=32, qfile=qfile, rfile=rfile)

#save
if qfile is None: cpool.save_qkernel(cl, saveqfile)
if rfile is None: cpool.save_rkernel(cl, saverfile)
cpool.save_qfunc(cl, '../data/RunPB/PTdata/q_func.dat')


#pk


for zz in np.arange(0., 3.5, 0.1):
    print('For z = %0.2f'%zz)
    pk= cpool.make_table(cl, nk=100, npool=32, M=0.2922, z=zz)
    cpool.save_pk(pk, '../data/RunPB/PTdata/pkcleft_zz%03d.dat'%(zz*100))

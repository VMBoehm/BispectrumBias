import numpy as np
import pk_pregen as PK

db = '../data/RunPB/PTdata/'

pk = PK.PkZCLEFT(zmin=1.5, zmax=2.5, db=db)

#bias signature is b1, b2, bs2, bn(0), alpha, sn, auto(True/False)
bias = [2.0, 1.0, 0., 0., 0., 0.]
nobias = [0.0, 0.0, 0., 0., 0., 0.]


#Returns a tuple of (k, pk)
pkhmz = lambda z: pk([z] + bias, auto=False)
pkmmz = lambda z: pk([z] + nobias, auto=False)

print(*bias)
print(pkhmz(z=1.8))
print(pkmmz(z=1.8))


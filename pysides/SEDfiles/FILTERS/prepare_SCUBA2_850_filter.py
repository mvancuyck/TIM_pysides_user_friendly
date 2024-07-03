import pandas as pd
import numpy as np
from astropy import constants as const
import astropy.units as u
from IPython import embed
import matplotlib.pyplot as plt 

data = np.array(pd.read_csv('SCUBA2_model850.txt', sep = '\s'))

wl = np.flip((const.c / (data[:,0] * u.GHz)).to('um'))
transm = np.flip(data[:,15])

#plt.plot(wl,transm)

txt = ''

for k in range(0, len(wl)):
    txt += '{} {}\n'.format(wl.value[k], transm[k])

with open('SCUBA2_850.DAT', 'w') as file:
    file.write(txt)

embed()

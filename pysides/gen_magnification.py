import numpy as np
import pandas as pd
import scipy as sp
from time import time
from astropy.utils.console import ProgressBar

def gen_magnification(cat, params, magnify = True):

    tstart = time()

    print("Generate magnification...")

    if magnify == True:

        #Load the magnification grid
        data = np.array(np.loadtxt(params['path_mu_file'], delimiter = ','))
        z_grid = data[0,1:]
        mu_grid = np.flip(data[1:,0], 0)
        Psupmu = np.flip(data[1:,1:], axis = 0)
        indz_grid = np.arange(0,np.size(z_grid))
    
        indz_gal = np.fix(np.rint(np.interp(cat["redshift"], z_grid, indz_grid))) #no need to put condition for below 0 or above Nz value, since python assume a constant outside of the interpolated range
        indz_set = list(set(indz_gal))

    
        Ngal = np.shape(cat)[0]
    
        mu = np.zeros(Ngal)
        for k in ProgressBar(range(0, len(indz_set))):
            index = np.where(indz_gal == k)
            Xuni = np.random.rand(len(index[0]))
            mu[index[0]] = np.interp(Xuni, Psupmu[:,int(indz_set[k])], mu_grid)

    else:

        print('The magnify keyword is set to False. mu = 1 for all the sources.')
        mu = np.ones(len(cat))

    cat = cat.assign(mu = mu)
        
    tstop = time()

    print(len(cat), 'magnifications generated in ', tstop-tstart, 's')

    return cat

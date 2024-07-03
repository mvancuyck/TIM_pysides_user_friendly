import numpy as np
import pandas as pd
from astropy.cosmology import Planck15 as cosmo
from time import time
from astropy.utils.console import ProgressBar

from IPython import embed #for debugging purpose only

def gen_mass(params, zmin = 0., zmax = 11.):

    tstart = time()

    print('Generate stellar masses of galaxies for an unclustered catalog...')

    omega = ( np.pi / 180. )**2 * params['field_size'] #Field size converted to sr

    #If the field size is big, increase the redshift resolution to avoid steps in the redshift distribution
    Nlog1plusz = params['Nlog1plusz']
    dlog1plusz = params['dlog1plusz']
    if params['field_size'] > 10.:
        zgridfactor = np.sqrt(params['field_size']/10.) #Determine empirically to avoid statistically significant steps in the mass function
        Nlog1plusz = Nlog1plusz * zgridfactor
        dlog1plusz = dlog1plusz * 1./zgridfactor

    #Built the redshift grid
    log1plusz = dlog1plusz*(np.arange(0,Nlog1plusz+1)) #list of bins limit
    log1pluszgrid = dlog1plusz*(np.arange(0,Nlog1plusz)+0.5) #center of the bins (used to compute the analytical mass function in the bin, assumed to be not varying in the bin)
    zbins = 10.**log1plusz-1. #limit of the zbins (used to compute the redshift)
    zgrid = 10.**log1pluszgrid-1. #middle of the bins (used to estimate the mass function)
    Nz = np.size(zgrid)
    Dlum_bins = cosmo.luminosity_distance(zbins)
    Vbins = omega * np.diff(1. / 3. * ( Dlum_bins / (1.+zbins) )**3 ) #in Mpc^3

    #Build the mass grid
    Npts_mass = (params['logMmax']-params['logMcut']) * params['Npts_Mstar_dex']
    Mstargrid = 10.**(params['logMcut'] + (params['logMmax'] - params['logMcut']) / Npts_mass * np.arange(0, Npts_mass+1))
    dlogMstargrid = 1. / params['Npts_Mstar_dex']

    #load the evolution of the mass function
    (zmean, logMknee, phiknee1, phiknee2, alpha1, alpha2) = np.loadtxt(params['path_smf_evo_file'], unpack = True)

    inzrange = np.where( (zgrid >= zmin) & (zgrid <= zmax) )[0] #use just the bins, which has to be computed

    Mknee_z = 10.**np.interp(1.+zgrid[inzrange], 1.+zmean, logMknee)
    phiknee1_z = np.interp(1.+zgrid[inzrange], 1.+zmean, phiknee1)
    alpha1_z = np.interp(1.+zgrid[inzrange], 1.+zmean, alpha1)
    alpha2_z =  np.interp(1.+zgrid[inzrange], 1.+zmean, alpha2)
    
    phiknee2_z = np.exp(np.interp(np.log(1.+zgrid[inzrange]), np.log(1.+zmean), np.log(phiknee2)))
    if zmax > zmean[-1]:
        index_extrapol = np.where(zgrid[inzrange] > zmean[-1])
        slope = (np.log(phiknee2[-1])-np.log(phiknee2[-2])) / (zmean[-1] - zmean[-2])
        phiknee2_z[index_extrapol] = np.exp( np.log(phiknee2[-1]) + slope * (zgrid[inzrange[index_extrapol[0]]] - zmean[-1]))


    Mstar = np.zeros(0)
    redshift = np.zeros(0)
        
    for k in ProgressBar(inzrange):
        
        phi = np.exp(-Mstargrid/Mknee_z[k]) * (phiknee1_z[k]*(Mstargrid/Mknee_z[k])**alpha1_z[k] + phiknee2_z[k]*(Mstargrid/Mknee_z[k])**alpha2_z[k]) / Mknee_z[k] * Mstargrid * np.log(10)
        
        NsupM = np.array(np.cumsum(np.flip(phi * dlogMstargrid, 0)) * Vbins[k]) #Be careful, it starts from the highest mass to the lowest, the corresponding index are np.flip(Mstargrid)
        Ngal = np.random.poisson(lam = NsupM[-1], size = 1)[0]

        if Ngal > 0:
            Xuni = np.random.rand(Ngal)
            Mstar = np.concatenate( ( Mstar, np.interp( Xuni, NsupM / NsupM[-1] , np.flip(Mstargrid, 0))) )
            redshift = np.concatenate( ( redshift, zbins[k] + (zbins[k+1] - zbins[k]) * np.random.rand(Ngal) ) )
            
    Mstar = np.reshape(Mstar, np.size(Mstar))
    redshift = np.reshape(redshift, np.size(redshift))

    #Export in a pandas dataframe
    data = list(zip(redshift,Mstar))
    cat = pd.DataFrame(data = data, columns=['redshift', 'Mstar'])

    tstop = time()

    print(len(cat), 'galaxy masses generated in ', tstop-tstart, 's')

    return cat

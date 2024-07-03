import numpy as np
import pickle
from astropy.cosmology import Planck15 as cosmo

from astropy import constants as const
from astropy import units as u

from astropy.utils.console import ProgressBar

from IPython import embed #for debug only!

#Couple of standard parameters to prepare the grids#

Nlog1plusz = 1200  #Number of point at which the SMF is estimated (also number of bins)
dlog1plusz = 0.001 #size the of redshift bins in log(1+z) unit

##### The filter file, which path is provided as input of the code, must be a plain text with 2 columns: lambda in micron, t_nu the tranmission.

def gen_grid(path_filter, filter_name, renorm_lambda = 1.):

    print('Generate flux grids for the ', filter_name, ' filter (', path_filter,')...')
    
    #load the SEDs and define the redshift grid

    log1pluszgrid = dlog1plusz*(np.arange(1,Nlog1plusz+1))

    z_grid = 10.**log1pluszgrid-1

    N_z = np.size(z_grid)

    Dlum_grid = cosmo.luminosity_distance(z_grid)

    
    #Load the filters

    wl_filter, tnu_filter = np.loadtxt(path_filter, unpack = True)

    wl_filter = renorm_lambda * wl_filter

    ind=~np.isinf(wl_filter)	#discarding the infinite values
    wl_filter=wl_filter[ind]
    tnu_filter=tnu_filter[ind]

    nu_filter = np.array((const.c / ( wl_filter * u.um )).to('Hz').value)

    dnu_filter = np.abs(np.concatenate(([nu_filter[0]/2] , nu_filter[1:-1]-nu_filter[0:-2], [nu_filter[-1]/2])))

    norm_filter = 1 / np.sum(tnu_filter * dnu_filter)

    tnu_dnu_filter = np.matmul(np.expand_dims(tnu_filter * dnu_filter, 1), np.zeros([1,N_z])+1.) 

    #Load the SEDs

    SED_dict = pickle.load(open('SED_finegrid_dict.p', 'rb'))
    
    N_Umean = np.size(SED_dict['Umean'])

    #Loop over MS and SB templates

    type_list = ['MS', 'SB']

    filter_grid_dict = {}

    filter_grid_dict['dU'] = SED_dict['dU']
    filter_grid_dict['Umean'] = SED_dict['Umean']
    filter_grid_dict['Nlog1plusz'] = Nlog1plusz
    filter_grid_dict['dlog1plusz'] = dlog1plusz
    

    #compute the rest frequency for each point of the filter curve and each z 

    lambda_rest_filter_arr = np.zeros([len(wl_filter),N_z])

    conv_factor_arr =  np.zeros([len(wl_filter),N_z]) #factor to multiply luminosity in nuLnu with to the Snu/LIR in Jy

    for k in range(0, len(z_grid)):

        lambda_rest_filter_arr[:,k] = wl_filter / (1 + z_grid[k])

        nu_rest = (const.c / ( lambda_rest_filter_arr[:,k] * u.um )).to('Hz')

        conv_factor_arr[:,k] = ((1 + z_grid[k]) / (4 * np.pi * Dlum_grid[k]**2 * nu_rest) * u.Lsun).to('Jy')

    #compute Snu/LIR for all the grid!

    for sed_type in type_list:

        print(sed_type, ' templates:' )

        Snu_LIR_grid = np.zeros([N_z, N_Umean])

        #Loop only over the Umean, the computation over z is made using an array with a redshift dimension (much faster because done in numpy and not in a for loop)

        for i in ProgressBar(range(0, N_Umean)):

            nuLnu_filter_arr = np.interp(lambda_rest_filter_arr, SED_dict['lambda'], SED_dict['nuLnu_'+sed_type+'_arr'][i,:])

            Snu_LIR_grid[:,i] = np.sum(nuLnu_filter_arr * conv_factor_arr * tnu_dnu_filter, axis = 0) * norm_filter
        
        filter_grid_dict['Snu_LIR_'+sed_type] = Snu_LIR_grid
    
    #Save the results!

    pickle.dump(filter_grid_dict, open('GRID_FILTER/'+filter_name+'.p', 'wb'))
    
    return True
    
    

    

    

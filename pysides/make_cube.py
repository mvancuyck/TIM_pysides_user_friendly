from pysides.gen_fluxes import gen_Snu_arr
from astropy.io import fits
import astropy.units as u
import scipy.constants as cst
import numpy as np
from astropy import wcs
from astropy.stats import gaussian_fwhm_to_sigma
import astropy.convolution as conv
import datetime
from astropy.table import Table
import pickle
from copy import deepcopy
import os
from IPython import embed

def gen_radec(cat, params):
#Routine used to generate ra and dec. Used only if the field size is provided, but not the ra and dec (catalog without clustering generated by gen_mass)
    
    n = int(len(cat['redshift']))

    ra_max = np.sqrt(0.1)
    dec_max =  np.sqrt(0.1) 

    ra = np.random.uniform(low=0, high=ra_max, size=(n,)) * u.deg
    dec = np.random.uniform(low=0, high=dec_max, size=(n,)) * u.deg

    return ra, dec 

def set_wcs(cat,params):

    #-----------------SET WCS-----------------
    # Coordinate increment at reference point
    if ((not "ra" in cat.columns) or (not "dec" in cat.columns)):
        print("generating the coordinates of the sources")
        ra,dec = gen_radec(cat, params )
    else:
        ra =  np.asarray(cat["ra"]) * u.deg
        dec = np.asarray(cat["dec"]) * u.deg
    
    ra_mean, dec_mean = np.mean(ra.value) , np.mean(dec.value) 

    # set a first time the wcs
    pix_resol = params["pixel_size"] / 3600.

    ra_cen = 0.5 * (ra.max() + ra.min())
    dec_cen = 0.5 * (dec.max() + dec.min())
    delta_ra = ra.max() - ra.min()
    delta_dec = dec.max() - dec.min()

    w = wcs.WCS(naxis=3)
    w.wcs.crval = [ra_cen.value, dec_cen.value, params["freq_min"]]
    w.wcs.crpix = [0.5*delta_ra.value / pix_resol, 0.5*delta_dec.value / pix_resol, 1]
    w.wcs.cdelt = [pix_resol, pix_resol, params["freq_resol"]]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN", "FREQ"]
    w.wcs.cunit = [u.deg, u.deg, u.Hz]

    # list of the position in pixel units from wcs: pos_wcs[0] = x, pos_wcs[1] = y
    x, y = w.celestial.wcs_world2pix(ra , dec , 0)

    #Offset the central pixel to have all x>=0 and y>=0
    w.wcs.crpix = [0.5*delta_ra.value / pix_resol - np.min(x), 0.5*delta_dec.value / pix_resol  - np.min(y), 1]

    #recompute x and y in the new WCS
    x, y = w.celestial.wcs_world2pix(ra , dec, 0)

    # list of the position in pixel units: pos[0] = y, pos[1] = x 
    pos = [y , x]

    # compute the size of the frequencies axis
    zmax = w.swapaxes(0, 2).sub(1).wcs_world2pix(params['freq_max'], 0)[0]

    # compute the dimensions of the three axes
    shape = [np.ceil(zmax).astype(int), np.ceil(pos[0].max()), np.ceil(pos[1].max())]
    shape = [i//2*2+1 for i in shape] #force an odd number of pixels to generate better psf
    x_edges = list(np.arange(-0.5, shape[2] + 0.5, 1)) 
    y_edges = list(np.arange(-0.5, shape[1] + 0.5, 1)) 
    z_edges = list(np.arange(-0.5, shape[0] + 0.5, 1))

    wcs_dict = {}
    wcs_dict['w'] = w
    wcs_dict['shape'] = shape
    wcs_dict['pos'] = pos
    wcs_dict['x_edges'] = np.array(x_edges)
    wcs_dict['y_edges'] = np.array(y_edges)
    wcs_dict['z_edges'] = np.array(z_edges)

    return wcs_dict

def set_kernel(params, cube_prop_dict):

    #Produce a list of Nchannel convolution kernels corresponding to the beam in each channel

    kernel = []
    beam_area_pix2 = []

    z = np.arange(0,cube_prop_dict['shape'][0],1)
    w = cube_prop_dict['w']
    freq_list = w.swapaxes(0, 2).sub(1).wcs_pix2world(z, 0)[0]
    
    for freq in freq_list:
        fwhm = ((1.22 * cst.c) / (freq * params["telescop_diameter"])) * u.rad
        #print(freq, fwhm.to('arcsec'))
        sigma = (fwhm * gaussian_fwhm_to_sigma).to(u.arcsec)
        sigma_pix = sigma.value / params["pixel_size"]  #pixel
        kernel_channel = conv.Gaussian2DKernel(x_stddev=sigma_pix, x_size=cube_prop_dict['shape'][1])
        kernel_channel.normalize(mode="peak")
        kernel.append(kernel_channel)
        #size of the beam in pix2
        beam_area_pix2.append(np.sum(kernel_channel.array))
        
    return kernel, np.array(beam_area_pix2)

def save_cube(output_path, run_name, component_name, cube_type, cube_prop_dict, unit, input_cat, save_smoothed=True,cube=None, 
              recompute = False): 

    filename = output_path +'/'+ run_name + '_' + component_name + '_' +cube_type + '.fits'
    if(not os.path.isfile(filename) or recompute):
        print('Write '+filename+'...')
        f = fits.PrimaryHDU(cube, header=cube_prop_dict['w'].to_header())
        hdu = fits.HDUList([f])
        hdr = hdu[0].header
        hdr.set("cube")
        hdr.set("Datas")
        hdr["BITPIX"] = ("64", "array data type")
        hdr["BUNIT"] = unit
        hdr['COMMENT'] = f'Input catalog = {input_cat}, '
        hdr["DATE"] = (str(datetime.datetime.now()), "date of the creation")
        hdu.writeto(filename, overwrite=True)
        hdu.close()

        if(save_smoothed):

            for f in range(0, cube_prop_dict['shape'][0]):
                cube[f,:,:] = conv.convolve_fft(cube[f,:,:], cube_prop_dict['kernels'][f], 
                                                normalize_kernel=True, boundary="wrap")
            filename = output_path +'/'+ run_name + '_' + component_name + '_' +cube_type + '_smoothed.fits'
            f = fits.PrimaryHDU(cube, header=cube_prop_dict['w'].to_header())
            hdu = fits.HDUList([f])
            hdr = hdu[0].header
            hdr.set("cube")
            hdr.set("Datas")
            hdr["BITPIX"] = ("64", "array data type")
            hdr["BUNIT"] = unit
            hdr['COMMENT'] = f'Input catalog = {input_cat}, '
            hdr["DATE"] = (str(datetime.datetime.now()), "date of the creation")
            hdu.writeto(filename, overwrite=True)
            hdu.close()
            print('Write '+filename+'...')

        return 0
    else:
        cube=fits.getdata(filename)
        hdr = fits.getheader(filename)
        return 0 # cube, hdr

def save_cubes(cube_input, cube_prop_dict, params_sides, params, component_name, just_save = False, just_compute = False):

    #In the standard mode, cube_input should be the no beam cube in Jy/pix
    #The just_save == True option takes a dictionnary of cubes as cube_input (used for cubes combined using several components for save_cubes has already been run) and save only the cubes in the cube_inputs.
    #If just_compute is True, it will compute the convolved cubes, but will not write the result as a FITS file. 

    units_dict = {'nobeam_Jy_pix': 'Jy/pixel',
                  'nobeam_MJy_sr': 'MJy/sr',
                  'smoothed_Jy_beam': 'Jy/beam',
                  'smoothed_MJy_sr': 'MJy/sr'}
    
    if just_save == True:

        cubes2save = list(cube_input.keys())

        #By default cubes2save is always kept in memory in the result dictionnary even if the it is not saved on the disk. It thus appears in the cube_input keys. Removed if the save keyword is not True!
        if params['save_cube_nobeam_Jy_pix'] == False:
            cubes2save.remove('nobeam_Jy_pix')
        
        cubes_dict = cube_input
        
    else:

        cubes2save = []
        
        cubes_dict = {'nobeam_Jy_pix': cube_input}

        if params['save_cube_nobeam_Jy_pix']:
            cubes2save.append('nobeam_Jy_pix')

        if params['gen_cube_nobeam_MJy_sr']:
            pixel_sr = (params['pixel_size'] * np.pi/180/3600 )**2 #solid angle of the pixel in sr 
            cubes_dict['nobeam_MJy_sr'] =  cube_input / pixel_sr * 1.e-6 #The 1.e-6 is used to go from Jy to My, final result in MJy/sr
            cubes2save.append('nobeam_MJy_sr')

        if params['gen_cube_smoothed_Jy_beam'] or params['gen_cube_smoothed_MJy_sr']:
            print('Smooth the '+component_name+' cube by the beam...')
            smoothed_Jybeam = deepcopy(cube_input)
            for f in range(0, cube_prop_dict['shape'][0]):
                pixel_sr = (params['pixel_size'] * (np.pi/180/3600))**2 #solid angle of the pixel in sr 
                smoothed_Jybeam[f,:,:] = conv.convolve_fft( smoothed_Jybeam[f,:,:], cube_prop_dict['kernel'][f], normalize_kernel=False, boundary="wrap")
            cubes_dict['smoothed_Jy_beam'] =  smoothed_Jybeam
            
            if params['gen_cube_smoothed_Jy_beam']:
                cubes2save.append('smoothed_Jy_beam')

            if params['gen_cube_smoothed_MJy_sr']:
                cubes_dict['smoothed_MJy_sr'] = smoothed_Jybeam / (cube_prop_dict['beam_area_pix2'][:, np.newaxis, np.newaxis] * (params['pixel_size'] * np.pi/180/3600)**2) * 1.e-6            
                cubes2save.append('smoothed_MJy_sr')

    if not just_compute:
        for cube_type in cubes2save:

            filename = params["output_path"] + '/' +  params["run_name"] + '_' + component_name + '_' +cube_type + '.fits'
            
            print('Write '+filename+'...')
            
            if os.path.exists(params['output_path']) == False:
                print('Create '+params['output_path'])
                os.makedirs(params['output_path'])
            
            f = fits.PrimaryHDU(cubes_dict[cube_type], header=cube_prop_dict['w'].to_header())
            hdu = fits.HDUList([f])
            hdr = hdu[0].header
            hdr.set("cube")
            hdr.set("Datas")
            hdr["BITPIX"] = ("64", "array data type")
            hdr["BUNIT"] = units_dict[cube_type]
            hdr['COMMENT'] = 'telescope diameter = '+str(params['telescop_diameter'])+'m'
            hdr['COMMENT'] = 'Input catalog = '+params['sides_cat_path']
            hdr["DATE"] = (str(datetime.datetime.now()), "date of the creation")
            hdu.writeto(filename, overwrite=True)
            hdu.close()
    
    return cubes_dict

def channel_flux_densities(cat, params_sides, cube_prop_dict, params, filter=False,spc=10):

    z = np.arange(0,cube_prop_dict['shape'][0],1)
    w = cube_prop_dict['w']
    channels = w.swapaxes(0, 2).sub(1).wcs_pix2world(z, 0)[0]
    SED_dict = pickle.load(open(params_sides['SED_file'], "rb"))
    print("Generate CONCERTO monochromatic fluxes...")
    if(not filter): lambda_list =  ( cst.c * (u.m/u.s)  / (np.asarray(channels) * u.Hz)  ).to(u.um).value
    else:

        # Compute N-sigma range for each channel, N is given by params['freq_width_in_sigma']
        fwhm = w.wcs.cdelt[2] * gaussian_fwhm_to_sigma # Frequency resolution (step between consecutive channels)
        sigma = fwhm * gaussian_fwhm_to_sigma # Convert FWHM to sigma
        lower_bounds = channels - params['freq_width_in_sigma']/2 * sigma 
        upper_bounds = channels + params['freq_width_in_sigma']/2 * sigma 
        freq_list = np.linspace(lower_bounds.min(), upper_bounds.max(), spc*len(channels) )
        lambda_list = ( cst.c * (u.m/u.s)  / (np.asarray(freq_list) * u.Hz)  ).to(u.um).value
        
    Snu_arr = gen_Snu_arr(lambda_list, SED_dict, cat["redshift"], cat['mu']*cat["LIR"], cat["Umean"], cat["Dlum"], cat["issb"])

    if(filter):

        mask = (freq_list[:,np.newaxis] >= lower_bounds) & (freq_list[:,np.newaxis] < upper_bounds)
        transmission = np.exp(-((freq_list[:,np.newaxis]- channels) ** 2) / (2 * (sigma)**2)) 
        Snu_arr_transmitted = Snu_arr[:,:,np.newaxis] * mask.astype(int)* transmission 
        freq_transmitted = freq_list[:,np.newaxis]*mask
        Snu_transmitted = np.sum(Snu_arr_transmitted , axis=1)
        Snu_arr = Snu_transmitted
            
    return Snu_arr

def make_continuum_cube(cat, params_sides, params, cube_prop_dict, filter=False):

    continuum_nobeam_Jypix = []

    channels_flux_densities = channel_flux_densities(cat, params_sides,cube_prop_dict, params, filter=filter)
    for f in range(0, cube_prop_dict['shape'][0]):      
        row = channels_flux_densities[:,f] #Jy/pix
        histo, y_edges, x_edges = np.histogram2d(cube_prop_dict['pos'][0], cube_prop_dict['pos'][1], bins=(cube_prop_dict['y_edges'], cube_prop_dict['x_edges']), weights=row)
        continuum_nobeam_Jypix.append(histo) #Jy/pix, no beam

    embed()

    continuum_cubes = save_cubes(np.asarray(continuum_nobeam_Jypix), cube_prop_dict, params_sides, params, 'continuum', just_compute = not params['save_continuum_only'])
    
    return continuum_cubes

def line_channel_flux_densities(line, rest_freq, cat, cube_prop_dict):

    freq_obs = rest_freq / (1 + cat['redshift']) #GHz
    #freq_channel = np.round(freq_obs,0).astype(int) #GHz
    channel = np.asarray(cube_prop_dict['w'].swapaxes(0, 2).sub(1).wcs_world2pix(freq_obs*1e9, 0))[0] 
    nudelt = abs(cube_prop_dict['w'].wcs.cdelt[2]) * 1e-9 #GHz
    vdelt = (cst.c * 1e-3) * nudelt / freq_obs #km/s
    S = cat['I'+line] / vdelt  #Jy
    
    return S, channel

def line_filter_flux_densities(line, rest_freq, cat, cube_prop_dict,params):
    """
    Find the overlapping spectral channels for an array of observed frequencies.

    Parameters:
    - w : astropy.wcs.WCS
        The WCS object with a frequency axis.
    - f_obs_array : array-like
        Array of observed frequencies (in Hz).
    - gaussian_fwhm_to_sigma : float
        Conversion factor from FWHM to sigma.
    - n_channels : int
        Number of spectral channels.

    Returns:
    - list of lists of int
        List of indices of channels for each frequency in `f_obs_array`.
    """

    z = np.arange(0,cube_prop_dict['shape'][0],1)
    w = cube_prop_dict['w']
    freq_list = w.swapaxes(0, 2).sub(1).wcs_pix2world(z, 0)[0]
    
    # Compute N-sigma range for each channel, N is given by params['freq_width_in_sigma']
    fwhm = w.wcs.cdelt[2] * gaussian_fwhm_to_sigma # Frequency resolution (step between consecutive channels)
    sigma = fwhm * gaussian_fwhm_to_sigma # Convert FWHM to sigma
    lower_bounds = freq_list - params['freq_width_in_sigma']/2 * sigma 
    upper_bounds = freq_list + params['freq_width_in_sigma']/2 * sigma 

    #test_indices = [89938, 1388638, 10271987]
    freq_obs = np.asarray(rest_freq / (1 + cat['redshift']) )#GHz
    nudelt = abs(cube_prop_dict['w'].wcs.cdelt[2]) / 1e9 #GHz
    vdelt = (cst.c * 1e-3) * nudelt / freq_obs #km/s
    Snu = np.asarray(cat['I'+line] / vdelt)  #Jy

    # Broadcasting: Check which channels contain each observed frequency
    mask = (freq_obs[:, np.newaxis] >= lower_bounds/1e9) & (freq_obs[:, np.newaxis] < upper_bounds/1e9)
    channels_list = np.where(mask, np.arange(len(freq_list)), -2)
    #----
    #Gaussian spectral profile
    transmission = np.exp(-((freq_obs[:, np.newaxis] - freq_list/1e9) ** 2) / (2 * (sigma/1e9)**2)) 

    #Lorentz spectral profile 
    #transmission = 1/(((freq_obs[:, np.newaxis]-freq_list/1e9)/(fwhm/1e9/2))**2 + 1 )
    #----
    Snu_transmitted = Snu[:,np.newaxis] * transmission * mask.astype(int)
    # Get indices of overlapping channels for each freq_obs
    #indices = np.where(mask)
    
    # Flatten channels_list and repeat pos accordingly
    channels_flat = channels_list.ravel()
    Snu_flat = Snu_transmitted.ravel()
    
    if((len(cube_prop_dict['pos'][0]) != len(freq_obs)*cube_prop_dict['shape'][0]) or (len(cube_prop_dict['pos'][1]) != len(freq_obs)*cube_prop_dict['shape'][0])):
        x_flat = np.repeat(cube_prop_dict['pos'][1], cube_prop_dict['shape'][0])  # Repeat each source position
        y_flat = np.repeat(cube_prop_dict['pos'][0], cube_prop_dict['shape'][0])
        cube_prop_dict['pos'] = (y_flat, x_flat)

    #Check only spectral axis
    #hist, _ = np.histogram(channels_flat, bins=cube_prop_dict['z_edges'], weights=Snu_flat)  

    return Snu_flat, channels_flat

def make_co_cube(cat, params_sides, params, cube_prop_dict,filter=False):

    #Create the cube for each line, save it and add it to the lines cube
    first_Jup = 1
    last_Jup = 8
    
    for J in range(first_Jup, last_Jup+1):
        line_name = "CO{}{}".format(J, J - 1)
        print('Compute channel locations and flux densities of '+line_name+' lines...')
        rest_freq = params_sides["nu_CO"] * J 
        if(not filter): Snu, channels = line_channel_flux_densities(line_name, rest_freq, cat, cube_prop_dict)
        else: Snu, channels = line_filter_flux_densities(line_name, rest_freq, cat, cube_prop_dict, params)

        print('Generate the non-smoothed '+line_name+' cube...')
        CO_oneJ_nobeam_Jypix, edges = np.histogramdd(sample=(channels, cube_prop_dict['pos'][0], cube_prop_dict['pos'][1]), bins=(cube_prop_dict['z_edges'], cube_prop_dict['y_edges'], cube_prop_dict['x_edges']), weights=Snu)
        CO_oneJ_cubes = save_cubes(CO_oneJ_nobeam_Jypix, cube_prop_dict, params_sides, params, line_name, just_compute = not params['save_each_transition'])

        if J == first_Jup:
            keys_computed_cubes = list(CO_oneJ_cubes.keys()) #list of type of cubes computed (smoothed or not, unit)
            CO_all_cubes = deepcopy(CO_oneJ_cubes)
        else:
            for key in keys_computed_cubes:
                CO_all_cubes[key] += CO_oneJ_cubes[key]

    if params['save_each_line']:  
        print('Save the CO cubes containing all the transitions...')
        save_cubes(CO_all_cubes, cube_prop_dict, params_sides, params, 'CO_all', just_save = True)

    return CO_all_cubes 

def make_cii_cube(cat, params_sides, params, cube_prop_dict, name_relation,filter=False):

    print('Compute channel locations and flux densities of [CII] line ('+name_relation+'et al.  recipe)...')
    if(not filter): Snu, channels = line_channel_flux_densities('CII_'+name_relation, params_sides["nu_CII"], cat, cube_prop_dict)
    else: Snu, channels = line_filter_flux_densities('CII_'+name_relation, params_sides["nu_CII"], cat, cube_prop_dict, params)
    
    print('Generate the non-smoothed [CII] cube...')
    CII_nobeam_Jypix, edges = np.histogramdd(sample=(channels, cube_prop_dict['pos'][0], cube_prop_dict['pos'][1]), bins=(cube_prop_dict['z_edges'], cube_prop_dict['y_edges'], cube_prop_dict['x_edges']), weights=Snu)

    CII_cubes = save_cubes(CII_nobeam_Jypix, cube_prop_dict, params_sides, params, 'CII_'+name_relation, just_compute = not params['save_each_line'])

    return CII_cubes

def make_fir_lines_cube(cat, params_sides, params, cube_prop_dict,filter=False):

    first_loop = True

    for line_name, line_nu in zip(params_sides['fir_lines_list'], params_sides['fir_lines_nu']):
        print('Compute channel locations and flux densities of ['+line_name+'] lines...')
        if(not filter): Snu, channels = line_channel_flux_densities(line_name, line_nu, cat, cube_prop_dict)
        else: Snu, channels = line_filter_flux_densities(line_name, line_nu, cat, cube_prop_dict, params)
        print('Generate the non-smoothed ['+line_name+'] cube...')
        FIR_1line_nobeam_Jypix, edges = np.histogramdd(sample=(channels, cube_prop_dict['pos'][0], cube_prop_dict['pos'][1]), bins=(cube_prop_dict['z_edges'], cube_prop_dict['y_edges'], cube_prop_dict['x_edges']), weights=Snu)

        FIR_1line_cubes = save_cubes(FIR_1line_nobeam_Jypix, cube_prop_dict, params_sides, params, line_name, just_compute = not params['save_each_line'])

    
        if first_loop:
            FIR_lines_cubes = deepcopy(FIR_1line_cubes)
            first_loop = False
        else:
            keys_computed_cubes = list(FIR_lines_cubes.keys()) #list of type of cubes computed (smoothed or not, unit)
            for key in keys_computed_cubes:
                FIR_lines_cubes[key] += FIR_lines_cubes[key]

    if params['save_each_line']:  
        print('Save the cubes containing all the far-IR lines...')
        save_cubes(FIR_lines_cubes, cube_prop_dict, params_sides, params, 'FIR_lines', just_save = True)
        
    
    return FIR_lines_cubes

def make_ci_cube(cat, params_sides, params, cube_prop_dict, filter=False):

    line_names = ['CI10', 'CI21']

    first_loop = True
    
    for line_name in line_names:
    
        print('Compute channel locations and flux densities of ['+line_name+'] lines...')
        if(not filter): Snu, channels = line_channel_flux_densities(line_name, params_sides["nu_"+line_name], cat, cube_prop_dict)
        else: Snu, channels = line_filter_flux_densities(line_name, params_sides["nu_"+line_name], cat, cube_prop_dict, params)
        
        print('Generate the non-smoothed ['+line_name+'] cube...')
        CI_one_trans_nobeam_Jypix, edges = np.histogramdd(sample=(channels, cube_prop_dict['pos'][0], cube_prop_dict['pos'][1]), bins=(cube_prop_dict['z_edges'], cube_prop_dict['y_edges'], cube_prop_dict['x_edges']), weights=Snu)

        CI_one_trans_cubes = save_cubes(CI_one_trans_nobeam_Jypix, cube_prop_dict, params_sides, params, line_name, just_compute = not params['save_each_transition'])

        if first_loop:
            CI_both_cubes = deepcopy(CI_one_trans_cubes)
            first_loop = False
        else:
            keys_computed_cubes = list(CI_both_cubes.keys()) #list of type of cubes computed (smoothed or not, unit)
            for key in keys_computed_cubes:
                CI_both_cubes[key] += CI_one_trans_cubes[key]

    if params['save_each_line']:  
        print('Save the [CI] cubes containing all the transitions...')
        save_cubes(CI_both_cubes, cube_prop_dict, params_sides, params, 'CI_both', just_save = True)

    return CI_both_cubes

def make_cube(cat ,params_sides, params_cube,filter=False):


    print("Set World Coordinates System...")
    cube_prop_dict = set_wcs(cat, params_cube)
    
    #compute the kernel
    if params_cube['gen_cube_smoothed_Jy_beam'] or params_cube['gen_cube_smoothed_MJy_sr']:
        print("Compute the beams for all channels...")
        cube_prop_dict['kernel'], cube_prop_dict['beam_area_pix2'] = set_kernel(params_cube, cube_prop_dict)
    
    print("Create continuum cubes..")                                                                              
    if(params_cube['save_continuum_only'] or params_cube['save_full']): 
        continuum_cubes = make_continuum_cube(cat, params_sides, params_cube, cube_prop_dict, filter=filter)

    print("Create CO cubes...")
    CO_cubes = make_co_cube(cat, params_sides, params_cube, cube_prop_dict, filter=filter)

    print("Create CI cubes...")
    CI_cubes = make_ci_cube(cat, params_sides, params_cube, cube_prop_dict, filter=filter)

    CII_relations_2compute = []

    keys_computed_cubes = list(CO_cubes.keys()) #if a type of cube is computed, then it is computed for all the components
    
    if (params_cube['gen_cube_CII_Lagache']):
        CII_relations_2compute.append('Lagache')
    if (params_cube['gen_cube_CII_de_Looze']):
        CII_relations_2compute.append('de_Looze')
    '''
    if params_cube['add_fir_lines']:
        print('add FIR lines...')
        FIR_lines_cubes = make_fir_lines_cube(cat, params_sides, params_cube, cube_prop_dict)
    '''
    for CII_relation_name in CII_relations_2compute:
        CII_cubes = make_cii_cube(cat, params_sides, params_cube, cube_prop_dict, CII_relation_name, filter=filter)

        #Compute and save the combined cubes

        if params_cube['save_all_lines'] or params_cube['save_full']:

            combined_cubes = deepcopy(CO_cubes)

            print('Generate the cube(s) with all the lines...' )
            for key in keys_computed_cubes:
                combined_cubes[key] += CII_cubes[key]
                combined_cubes[key] += CI_cubes[key]
                '''
                if params_cube['add_fir_lines']:
                    combined_cubes[key] += FIR_lines_cubes[key]
                '''
            if params_cube['save_all_lines']:
                print('Save the cube(s) with all the lines...')
                save_cubes(combined_cubes, cube_prop_dict, params_sides, params_cube, 'all_lines_'+CII_relation_name, just_save = True)

            if params_cube['save_full']:
                for key in keys_computed_cubes:
                    combined_cubes[key] += continuum_cubes[key]
                save_cubes(combined_cubes, cube_prop_dict, params_sides, params_cube, 'full_'+CII_relation_name, just_save = True)
            
    print("Done!")

    return True

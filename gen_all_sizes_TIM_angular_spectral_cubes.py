import sys
import os
from analysis_fcts.fcts import * 
from pysides.make_cube import *
from pysides.load_params import *
import argparse
import time
import matplotlib
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import re
import glob

freq_CII = 1900.53690000 * u.GHz

def sorted_files_by_n(directory, tile_sizes):
    # List all files in the directory
    files = os.listdir(directory)
    
    sorted_files = []
    
    for tile_sizeRA, tile_sizeDEC in tile_sizes:
        # Define the regex pattern to match the files and extract 'n'
        pattern = re.compile(f'pySIDES_from_uchuu_tile_(\d+)_({tile_sizeRA}deg_x_{tile_sizeDEC}deg)\.fits')
        
        # Create a list of tuples (n, filename)
        files_with_n = []
        for filename in files:
            match = pattern.match(filename)
            if match:
                n = int(match.group(1))
                files_with_n.append((n, filename))
        
        # Sort the list by the value of 'n'
        files_with_n.sort(key=lambda x: x[0])
        
        # Extract the sorted filenames
        sorted_filenames = [filename for n, filename in files_with_n]
        sorted_files.extend(sorted_filenames)
    
    return sorted_files

def gen_spatial_spectral_cube(cat, cat_name, pars):
    
    #The prefix used to save the outputs maps:
    res = pars['pixel_size'];    pixel_sr = (res * np.pi/180/3600)**2 #solid angle of the pixel in sr
    params_cube = {'freq_min':pars['freq_min'], 
                   'freq_max':pars['freq_max'], 
                   'freq_resol':pars['freq_resol'], 
                   'pixel_size':res}

    cube_prop_dict = set_wcs(cat, params_cube)
    cube_all_lines = np.zeros(cube_prop_dict['shape'])
    cube_CO = np.zeros(cube_prop_dict['shape'])
    cube_CI = np.zeros(cube_prop_dict['shape'])

    z = np.arange(0,cube_prop_dict['shape'][0],1)
    w = cube_prop_dict['w']
    freqs = w.swapaxes(0, 2).sub(1).wcs_pix2world(z, 0)[0]

    #------------------------------------------------------
    kernel = []
    beam_area_pix2 = []
    for freq in freqs:

        fwhm = ((1.22 * cst.c) / (freq * pars["telescop_diameter"])) * u.rad
        #print(freq, fwhm.to('arcsec'))
        sigma = (fwhm * gaussian_fwhm_to_sigma).to(u.arcsec)
        sigma_pix = sigma.value / pars["pixel_size"]  #pixel
        kernel_channel = conv.Gaussian2DKernel(x_stddev=sigma_pix, x_size=cube_prop_dict['shape'][1])
        kernel_channel.normalize(mode="peak")
        kernel.append(kernel_channel)
        #size of the beam in pix2
        beam_area_pix2.append(np.sum(kernel_channel.array))

    cube_prop_dict['kernels'] = kernel
    cube_prop_dict['beam_area'] = np.array(beam_area_pix2)
    #------------------------------------------------------

    for j, (line, rest_freq) in enumerate(zip(line_list, rest_freq_list)):
        #--- Creates the intrinsic line intensity map and save it ---#
        S, channels = line_channel_flux_densities(line, rest_freq, cat, cube_prop_dict)

        line_map, edges = np.histogramdd(sample=(channels, cube_prop_dict['pos'][0], cube_prop_dict['pos'][1]), 
                                         bins  =(cube_prop_dict['z_edges'], cube_prop_dict['y_edges'], cube_prop_dict['x_edges']),
                                         weights=S)

        line_map *= 1 / pixel_sr / 1e6
        if(pars['save_each_transition']): save_cube(pars['output_path'], cat_name, line, "MJy_sr", 
                  cube_prop_dict, 'MJy per sr', pars['output_path']+cat_name, pars['gen_cube_smoothed_MJy_sr'], cube=line_map)
                    
        if('CO' in line): cube_CO += line_map
        if(line == 'CO87' and pars['save_each_line']):  
            save_cube(pars['output_path'], cat_name, 'CO_all', "MJy_sr", 
                      cube_prop_dict, 'MJy per sr', pars['output_path']+cat_name, pars['gen_cube_smoothed_MJy_sr'], cube=cube_CO)
        if('CI' in line and 'CII' not in line): cube_CI += line_map
        if(line=='CI21' and pars['save_each_line']): 
            save_cube(pars['output_path'], cat_name, 'CI_both', "MJy_sr", 
                      cube_prop_dict, 'MJy per sr', pars['output_path']+cat_name, pars['gen_cube_smoothed_MJy_sr'], cube=cube_CI)
        if('CII' in line and pars['save_each_line']):
            save_cube(pars['output_path'], cat_name, line, "MJy_sr", 
                  cube_prop_dict, 'MJy per sr', pars['output_path']+cat_name, pars['gen_cube_smoothed_MJy_sr'], cube=line_map)

        cube_all_lines+= line_map
    
    if(pars['save_all_lines']): save_cube(pars['output_path'], cat_name, 'all_lines', "MJy_sr", 
                                    cube_prop_dict, 'MJy per sr', pars['output_path']+cat_name, pars['gen_cube_smoothed_MJy_sr'], cube=cube_all_lines)
    
    if(pars['save_continuum_only'] or pars['save_full']):
        lambda_list =  ( cst.c * (u.m/u.s)  / (np.asarray(freqs)*1e9 * u.Hz)  ).to(u.um)
        SED_dict = pickle.load(open('pysides/SEDfiles/SED_finegrid_dict.p', "rb"))
        print("Generate monochromatic fluxes...")
        Snu_arr = gen_Snu_arr(lambda_list.value, SED_dict, cat["redshift"], cat['mu']*cat["LIR"], cat["Umean"], cat["Dlum"], cat["issb"])
        histo, edges = np.histogramdd(sample=(channels,cube_prop_dict['pos'][0], cube_prop_dict['pos'][1]), bins=(cube_prop_dict['z_edges'], cube_prop_dict['y_edges'], cube_prop_dict['x_edges']), weights=Snu_arr[:, 0])
        continuum = histo.value / pixel_sr / 1e6
        if(pars['save_continuum_only']): save_cube(pars['output_path'], cat_name, 'continuum', "MJy_sr",  
                  cube_prop_dict, 'MJy per sr', pars['output_path']+cat_name, pars['gen_cube_smoothed_MJy_sr'], cube=continuum)
        if(pars['save_full']):
            full = cube_all_lines + continuum
            save_cube(pars['output_path'], cat_name, 'full', "MJy_sr",
                      cube_prop_dict, 'MJy per sr', pars['output_path']+cat_name, pars['gen_cube_smoothed_MJy_sr'], cube=full, )

    if(pars['save_galaxies']):
        #---Creates the corresponding galaxy map, with a stellar mass cut mstar_cut
        cat_galaxies = cat.loc[cat["Mstar"] >= pars['Mstar_lim']]
        x, y = cube_prop_dict['w'].celestial.wcs_world2pix(cat_galaxies['ra'] , cat_galaxies['dec'], 0)
        freq_obs = rest_freq/(1+cat_galaxies["redshift"])
        channels_gal = np.asarray(cube_prop_dict['w'].swapaxes(0, 2).sub(1).wcs_world2pix(freq_obs*1e9, 0))[0] 
        galaxy_cube, edges = np.histogramdd(sample=(channels_gal, y, x), 
                                            bins = (cube_prop_dict['z_edges'], cube_prop_dict['y_edges'], cube_prop_dict['x_edges']))
        save_cube(pars['output_path'], cat_name, 'galaxies', 'pix',
                  cube_prop_dict, 'pix', pars['output_path']+cat_name, False, cube=galaxy_cube) 

    return 0


if __name__ == "__main__":

    params_sides = load_params('PAR_FILES/SIDES_from_original_with_fir_lines.par')
    TIM_params = load_params('PAR_FILES/Uchuu_cubes_for_TIM.par')

    for tile_sizeRA, tile_sizeDEC in TIM_params['tile_sizes']: 
        print(tile_sizeRA, tile_sizeDEC)
        
        # List files matching the pattern
        files = sorted_files_by_n(TIM_params["output_path"], ((tile_sizeRA, tile_sizeDEC),))
        
        for l, file in enumerate(files):

            cat = Table.read(TIM_params["output_path"]+file)
            cat = cat.to_pandas()
    
            params_cube = load_params("PAR_FILES/Uchuu_cubes_for_TIM.par")
            params_cube['run_name'] = f"pySIDES_from_uchuu_TIM_tile{l}"

            make_cube(cat ,params_sides, params_cube)

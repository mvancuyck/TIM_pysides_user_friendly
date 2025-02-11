import sys
import os
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
from multiprocessing import Pool, cpu_count

def worker_init(*args):
    global _args
    _args = args

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

def worker_compute(params):
    global _args
    cat, params_sides = _args
    for profile, range, diff_nu in params:  

        TIM_params = load_params(f'PAR_FILES/Uchuu_minicubes_for_TIM_{range}.par')

        ##Generate the TIM cubes with params precised in TIM_params.par
        TIM_params['run_name'] = f"pySIDES_from_uchuu_TIM_tile{l}_{range}_{tile_sizeRA}deg_{tile_sizeDEC}deg_res{TIM_params['pixel_size']:.0f}arcsec_dnu{TIM_params['freq_resol']/1e9:.1f}GHz"
        file = TIM_params['output_path'] +  TIM_params['run_name'] + '_full_de_Looze_smoothed_MJy_sr.fits' 
        if(not os.path.isfile(file)): make_cube(cat, params_sides, TIM_params)

        TIM_params['profile'] = profile
        TIM_params['run_name'] = f"pySIDES_from_uchuu_{profile}_TIM_tile{l}_{range}_{tile_sizeRA}deg_{tile_sizeDEC}deg_res{TIM_params['pixel_size']:.0f}arcsec_dnu{TIM_params['freq_resol']/1e9:.1f}GHz_minus{TIM_params['diff_btw_freq_resol_and_fwhm']/1e9:.1f}GHz_forfwhm"
        file = TIM_params['output_path'] +  TIM_params['run_name'] + '_full_de_Looze_smoothed_MJy_sr.fits' 
        if(not os.path.isfile(file)): make_cube(cat, params_sides, TIM_params)

    return 0

def make_all_cubes(cat,params_sides, ncpus=24):

    profiles = ('gaussian','lorentzian')
    zrange = ('highz', 'lowz', 'midz')
    diff_nu_list = (0, 0.2e9, 0.4e9, 0.6e9, 0.8e9, 1e9)
    
    params_list = []
    #for n in n_list:
    for profile in profiles:
        for range in zrange:
            #for diff_nu in diff_nu_list:
            params_list.append(list((profile,range,0 )))

    print("start parallelization")
    with Pool(ncpus, initializer=worker_init, initargs=list((cat, params_sides))) as p:
        zero = p.map(worker_compute, np.array_split(params_list, ncpus) )
    return 0   

if __name__ == "__main__":

    params_sides = load_params('PAR_FILES/SIDES_from_original_with_fir_lines.par')
    TIM_params = load_params('PAR_FILES/Uchuu_cubes_for_TIM.par')
    CONCERTO_params = load_params('PAR_FILES/Uchuu_cubes_for_CONCERTO.par')
    
    if(True):
        for i, (tile_sizeRA, tile_sizeDEC) in enumerate(TIM_params['tile_size']): 
            
            # List files matching the pattern
            files = sorted_files_by_n(TIM_params["sides_cat_path"], ((tile_sizeRA, tile_sizeDEC),))
            
            for l, cfile in enumerate(files):
                #Load the catalog of the subfield
                cat = Table.read(TIM_params["sides_cat_path"]+cfile)
                cat = cat.to_pandas()
                
                #Generate the TIM cubes with params precised in TIM_params.par
                TIM_params['run_name'] = f"pySIDES_from_uchuu_TIM_tile{l}_{tile_sizeRA}deg_{tile_sizeDEC}deg_res{TIM_params['pixel_size']:.0f}arcsec_dnu{TIM_params['freq_resol']/1e9:.1f}GHz"
                file = TIM_params['output_path'] +  TIM_params['run_name'] + '_full_de_Looze_smoothed_MJy_sr.fits' 
                if(not os.path.isfile(file) or True): make_cube(cat, params_sides, TIM_params)
                

                #To uncomment later for the gif.
                #-------------------------------
                TIM_params['profile'] = 'gaussian'
                freqs = np.arange(TIM_params['freq_min'], TIM_params['freq_max']+TIM_params['freq_resol'], TIM_params['freq_resol'])
                nbin = 10
                bins_low = freqs[::nbin]
                bins_high = freqs[::nbin]+(nbin-1)*TIM_params['freq_resol']
                for j, (low, high) in enumerate(zip(bins_low, bins_high)):
                    TIM_params['run_name'] = f"pySIDES_from_uchuu_gaussian_TIM_tile{l}_bin{j+1}_over_{len(bins_low)}_{tile_sizeRA}deg_{tile_sizeDEC}deg_res{TIM_params['pixel_size']:.0f}arcsec_dnu{TIM_params['freq_resol']/1e9:.1f}GHz_minus{TIM_params['diff_btw_freq_resol_and_fwhm']/1e9:.1f}GHz_forfwhm"
                    TIM_params['freq_max'] = high
                    TIM_params['freq_min'] = low
                    file = TIM_params['output_path'] +  TIM_params['run_name'] + '_full_de_Looze_smoothed_MJy_sr.fits' 
                    if(not os.path.isfile(file) and l==0 and ((tile_sizeRA==0.2) or (tile_sizeRA==1.414))):  make_cube(cat, params_sides, TIM_params)
                #-------------------------------

                #Generate the CONCERTO cubes if wanted
                '''
                CONCERTO_params['run_name'] = f"pySIDES_from_uchuu_CONCERTO_tile{l}_{tile_sizeRA}deg_{tile_sizeDEC}deg_res{CONCERTO_params['pixel_size']}arcsec_dnu{CONCERTO_params['freq_resol']/1e9}GHz"
                file = CONCERTO_params['output_path'] +  CONCERTO_params['run_name'] + 'full_de_Looze_smoothed_MJy_sr.fits' 
                if(not os.path.isfile(file)): make_cube(cat, params_sides, CONCERTO_params)
                '''
    embed()
    #generate smaller chunks of the TIM cube
    for tile_sizeRA, tile_sizeDEC in TIM_params['tile_size']: 

        # List files matching the pattern
        files = sorted_files_by_n(TIM_params["sides_cat_path"], ((tile_sizeRA, tile_sizeDEC),))

        for zrange in ('highz', 'lowz', 'midz'):
            TIM_params = load_params(f'PAR_FILES/Uchuu_minicubes_for_TIM_{zrange}.par')

            for l, cfile in enumerate(files):

                if(l>9): continue

                #Load the catalog of the subfield
                cat = Table.read(TIM_params["sides_cat_path"]+cfile)
                cat = cat.to_pandas()

                ##Generate the TIM cubes with params precised in TIM_params.par
                TIM_params['run_name'] = f"pySIDES_from_uchuu_TIM_tile{l}_{zrange}_{tile_sizeRA}deg_{tile_sizeDEC}deg_res{TIM_params['pixel_size']:.0f}arcsec_dnu{TIM_params['freq_resol']/1e9:.1f}GHz"
                file = TIM_params['output_path'] +  TIM_params['run_name'] + '_full_de_Looze_smoothed_MJy_sr.fits' 
                if(not os.path.isfile(file) ): make_cube(cat, params_sides, TIM_params)

                for profile in ('gaussian',):
                    #for minusdnu in ():

                    TIM_params['profile'] = profile

                    TIM_params['run_name'] = f"pySIDES_from_uchuu_{profile}_TIM_tile{l}_{zrange}_{tile_sizeRA}deg_{tile_sizeDEC}deg_res{TIM_params['pixel_size']:.0f}arcsec_dnu{TIM_params['freq_resol']/1e9:.1f}GHz_minus{TIM_params['diff_btw_freq_resol_and_fwhm']/1e9:.1f}GHz_forfwhm"
                    file = TIM_params['output_path'] +  TIM_params['run_name'] + '_full_de_Looze_smoothed_MJy_sr.fits' 
                    if(not os.path.isfile(file) or True): make_cube(cat, params_sides, TIM_params)
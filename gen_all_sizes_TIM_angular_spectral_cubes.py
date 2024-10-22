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

if __name__ == "__main__":

    params_sides = load_params('PAR_FILES/SIDES_from_original_with_fir_lines.par')
    TIM_params = load_params('PAR_FILES/Uchuu_cubes_for_TIM.par')

    for tile_sizeRA, tile_sizeDEC in TIM_params['tile_sizes']: 
        
        # List files matching the pattern
        files = sorted_files_by_n(TIM_params["sides_cat_path"], ((tile_sizeRA, tile_sizeDEC),))
        
        for l, file in enumerate(files):
            
            cat = Table.read(TIM_params["sides_cat_path"]+file)
            cat = cat.to_pandas()
    
            params_cube = load_params("PAR_FILES/Uchuu_cubes_for_TIM.par")
            params_cube['run_name'] = f"pySIDES_from_uchuu_TIM_tile{l}_{tile_sizeRA}deg_{tile_sizeDEC}deg_res{TIM_params['pixelsize']}arcsec_dnu{TIM_params['freq_resol']/1e9}GHz"

            make_cube(cat, params_sides, params_cube)

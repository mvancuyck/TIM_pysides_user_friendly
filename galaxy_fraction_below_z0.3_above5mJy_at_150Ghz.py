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
from progress.bar import Bar


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
    TIM_params   = load_params('PAR_FILES/Uchuu_cubes_for_TIM.par')

    for tile_sizeRA, tile_sizeDEC in TIM_params['tile_sizes']: 
        
        # List files matching the pattern
        files = sorted_files_by_n(TIM_params["sides_cat_path"], ((tile_sizeRA, tile_sizeDEC),))

        list_fract = np.zeros(len(files))

        dict_name = f'dict_dir/galaxy_fraction_below_z.3_above5mJy_at_150GHz_in_{tile_sizeRA}degx{tile_sizeDEC}deg.p'
        if(not os.path.isfile(dict_name)):

            dict = {}
            bar = Bar(f'Processing {tile_sizeRA}degx{tile_sizeDEC}deg', max=len(files))

            for l, file in enumerate(files):

                dict[f'{l}'] = {}
                
                cat = Table.read(TIM_params["sides_cat_path"]+file)
                cat = cat.to_pandas()
                cat = cat.loc[cat['redshift']<=0.3]

                channels = (150e9,) #GHz
                treshold = 5e-3 #Jy 
                lambda_list =  ( cst.c * (u.m/u.s)  / (np.asarray(channels)* u.Hz)  ).to(u.um)
                SED_dict = pickle.load(open(params_sides['SED_file'], "rb"))    
                Snu_arr = gen_Snu_arr(lambda_list.value, 
                                    SED_dict, cat["redshift"],
                                    cat["LIR"]*cat['mu'], cat["Umean"], 
                                    cat["Dlum"], cat["ISSB"])
                
                dict[f'{l}']['Snu_arr'] = Snu_arr
                dict[f'{l}']['len_cat_z<=0.3'] = len(cat)
                a = np.where(Snu_arr[:,0].value>=treshold)
                dict[f'{l}']['nb_sources_above_treshold'] = len(a[0])

                list_fract[l] = len(a[0]) / len(cat)

                bar.next()
            dict['mean_frac'] = list_fract.mean()
            dict['std_frac'] = list_fract.std()
            bar.finish
            pickle.dump(dict, open(dict_name, 'wb'))
        
        else: dict =  pickle.load( open(dict_name, 'rb'))

    for tile_sizeRA, tile_sizeDEC in TIM_params['tile_sizes']: 

        dict_name = f'dict_dir/galaxy_fraction_below_z.3_above5mJy_at_150GHz_in_{tile_sizeRA}degx{tile_sizeDEC}deg.p'
        dict =  pickle.load( open(dict_name, 'rb'))
        print (f'The fraction in {tile_sizeRA}deg X {tile_sizeDEC}deg is: f={dict["mean_frac"]}+-{dict["std_frac"]}')
import sys
import os
import argparse
import time
import matplotlib
import numpy as np
import vaex as vx
from pysides.load_params import *
from pysides.gen_outputs import *

def load_cat():

    import matplotlib
    matplotlib.use("Agg")
    start = time.time()
    
    cats_dir_path='/data/SIDES/PYSIDES_UCHUU_OUTPUTS/vpeak_complete/'
    
    filenames=[]
    with os.scandir(cats_dir_path) as it:
        for entry in it:
            filenames.append(cats_dir_path+entry.name)
        filenames=np.array(np.unique(filenames,axis=0))
        
    print('Loading the total data frame ...')
    df = vx.open_many(filenames)

    print('Converting the dataframe to pandas df ...')
    cat = df.to_pandas_df(['redshift', 'ra', 'dec', 'SFR', 'issb', 'mu','Dlum', 'Umean', 'LIR', 'Mhalo','Mstar', 'issb', 
                           'LCII_de_Looze', 'LCII_Lagache','LprimCO10',
                           'ICO10', 'ICO21', 'ICO32', 'ICO43', 'ICO54', 'ICO65', 'ICO76', 'ICO87', 'ICII_de_Looze', 'ICI10', 'ICI21'])
    #cat =  Table.from_pandas(cat)
    
    end = time.time()
    timing = end - start
    print(f'Loaded in {np.round(timing,2)} sec!')

    cube_gal_params_file = "PAR_FILES/CONCERTO_uchuu_ref_gal_cubes.par"
    cube_params_file = "PAR_FILES/CONCERTO_uchuu_ref_cube.par"
    params_sides_file = 'PAR_FILES/SIDES_from_uchuu.par' 
    
    return 'pySIDES_from_uchuu', cat, cats_dir_path, int(117) #cube_gal_params_file, cube_params_file, params_sides_file, 

if __name__ == "__main__":

    #parser to choose the simu and where to save the outputs 
    #e.g: python gen_cubes_TIM_cubes_117deg2_uchuu.py 'outputs_uchuu/' 'uchuu'
    #will generate the 117deg2 SIDES-Uchuu maps around z+-dz/2 and saves them in outputs_uchuu

    params = load_params('PAR_FILES/SIDES_from_original_with_fir_lines.par')
    TIM_params = load_params('PAR_FILES/Uchuu_cubes_for_TIM.par')
    simu, cat, dirpath, fs = load_cat()

    #With SIDES Bolshoi, for rapid tests. 
    '''
    dirpath="/home/mvancuyck/"
    cat = Table.read(dirpath+'pySIDES_from_original.fits')
    cat = cat.to_pandas()
    simu='pySIDES_from_bolshoi'; fs=2
    '''

    for tile_sizeRA, tile_sizeDEC in TIM_params['tile_sizes']: 
        if(fs<tile_sizeRA*tile_sizeDEC): continue

        ragrid=np.arange(cat['ra'].min(),cat['ra'].max(),np.sqrt(tile_sizeRA))
        decgrid=np.arange(cat['dec'].min(),cat['dec'].max(),np.sqrt(tile_sizeDEC))

        grid=np.array(np.meshgrid(ragrid,decgrid))

        ra_index = np.arange(0,len(ragrid)-1,1)
        dec_index = np.arange(0,len(decgrid)-1,1)
        ra_grid, dec_grid = np.meshgrid(ra_index, dec_index)
        # Flatten the grids and stack them into a single array
        coords = np.stack((ra_grid.flatten(), dec_grid.flatten()), axis=1)

        for l, (ira, idec) in enumerate(coords):
            
            if l >= TIM_params['Nmax']: break 

            cat_subfield=cat.loc[(cat['ra']>=grid[0,idec,ira])&(cat['ra']<grid[0,idec,ira+1])&(cat['dec']>=grid[1,idec,ira])&(cat['dec']<grid[1,idec+1,ira])]
            params['run_name'] = f'{simu}_tile_{l}_{tile_sizeRA}deg_x_{tile_sizeDEC}deg'
            gen_outputs(cat_subfield, params)

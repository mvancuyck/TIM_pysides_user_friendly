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

def gen_comoving_cube(cat, cat_name, pars, line, rest_freq, z_center=6, Delta_z=0.5):

    output_name = f'{pars["output_path"]}/{cat_name}_cube_3D_z{z_center}_Jy_sr_{line}.fits'
    if(not os.path.isfile(output_name)):

        cat = cat.loc[np.abs(cat['redshift']-z_center) <= Delta_z/2]

        nu_obs = rest_freq / (1+z_center)
        dz = ( pars['freq_resol'] / 1e9 ) * (1+z_center) / nu_obs.value
        z_bins = np.arange(z_center-Delta_z/2-dz/2, z_center+Delta_z/2+dz/2, dz)
        z_list = np.arange(z_center-Delta_z/2, z_center+Delta_z/2, dz)

        res = (pars['pixel_size'] *u.arcsec).to(u.deg).value

        ragrid_bins =np.arange(cat['ra'].min() -res/2,cat['ra'].max() +res/2,res)
        decgrid_bins=np.arange(cat['dec'].min()-res/2,cat['dec'].max()+res/2,res)

        angular_grid =  np.array(np.meshgrid(ragrid_bins,decgrid_bins))

        #compute the coordinates in the cube in comoving Mpc
        Dc_center = cosmo.comoving_distance(z_list).value
        ragrid =np.arange(cat['ra'].min() , cat['ra'].max(),res)
        decgrid=np.arange(cat['dec'].min(),cat['dec'].max(),res)
        ra_center = np.mean(ragrid)
        dec_center = np.mean(decgrid)
        ys = Dc_center * ( ragrid[:, np.newaxis]  - ra_center)  * (np.pi/180) * np.cos(np.pi/180*ragrid[:, np.newaxis])
        xs = Dc_center * ( decgrid[:, np.newaxis] - dec_center) * (np.pi/180)

        L = cat[f'I{line}'] * 1.04e-3 * cat['Dlum']**2 * rest_freq/(1+cat['redshift'])
        I = (L * (cst.c*1e-3) * 4.02e7 / (4*np.pi) / rest_freq.to(u.Hz) / cosmo.H(cat['redshift']))  # in Lsun / Mpc^2 * Mpc^2/Sr * Mpc/Hz

        cube_MJy_per_sr_per_Mpc, edges = np.histogramdd(sample = (np.asarray(cat['redshift']), cat['ra'], cat['dec']), 
                                                bins = (z_bins, ragrid_bins, decgrid_bins), weights = I)

        #convert to the proper unit (Jy/sr/Mpc3)
        transverse_res_list = []
        radial_res_list = []
        for i, z in enumerate(z_list):
            dv_voxel = np.abs(ys[0,i]-ys[1,i]) * np.abs(xs[0,i]-xs[1,i]) * (cosmo.comoving_distance(z+dz/2) - cosmo.comoving_distance(z-dz/2)).value
            cube_MJy_per_sr_per_Mpc[i,:,:] /= dv_voxel
            radial_res_list.append((cosmo.comoving_distance(z+dz/2) - cosmo.comoving_distance(z-dz/2)).value)
            transverse_res_list.append(np.sqrt(np.abs(ys[0,i]-ys[1,i]) * np.abs(xs[0,i]-xs[1,i])))
        mean_transverse_res = np.asarray(transverse_res_list).mean()
        mean_radial_res = np.asarray(radial_res_list).mean()

        #save the cube!

        f= fits.PrimaryHDU(cube_MJy_per_sr_per_Mpc)
        hdu = fits.HDUList([f])
        hdr = hdu[0].header
        hdr.set("cube")
        hdr.set("Datas")
        hdr["BITPIX"] = ("64", "array data type")
        hdr["BUNIT"] = 'MJy/sr'
        hdr["DATE"] = (str(datetime.datetime.now()), "date of the creation")

        for i, (vox_size, npix) in enumerate(zip(( mean_transverse_res, mean_transverse_res, mean_radial_res),
                                                ( len(ragrid),         len(decgrid),        len(z_list) ))):
            hdr[f"CDELT{int(i+1)}"] = vox_size
            hdr[f"CUNIT{int(i+1)}"] = 'Mpc'
            #hdr[f"NAXIS{int(i+1)}"] = npix

        hdu.writeto(output_name, overwrite=True)
        print('save '+output_name)
        hdu.close()

    output_name = f'{pars["output_path"]}/{cat_name}_cube_3D_z{z_center}_galaxies.fits'
    if(not os.path.isfile(output_name)):
        cat = cat.loc[cat['Mstar']>pars['Mstar_lim']]
        cube_g, edges = np.histogramdd(sample = (np.asarray(cat['redshift']), cat['ra'], cat['dec']), 
                                                bins = (z_bins, ragrid_bins, decgrid_bins))
        #save the cube!
        f= fits.PrimaryHDU(cube_g)
        hdu = fits.HDUList([f])
        hdr = hdu[0].header
        hdr.set("cube")
        hdr.set("Datas")
        hdr["BITPIX"] = ("64", "array data type")
        hdr["BUNIT"] = 'nb of gal'
        hdr["DATE"] = (str(datetime.datetime.now()), "date of the creation")
        for i, (vox_size, npix) in enumerate(zip(( mean_transverse_res, mean_transverse_res, mean_radial_res),
                                                ( len(ragrid),         len(decgrid),        len(z_list) ))):
            hdr[f"CDELT{int(i+1)}"] = vox_size
            hdr[f"CUNIT{int(i+1)}"] = 'Mpc'
            #hdr[f"NAXIS{int(i+1)}"] = npix
        hdu.writeto(output_name, overwrite=True)
        print('save '+output_name)
        hdu.close()

    return 0

    '''
    cube_voxel_address = np.zeros((len(z_list), len(ragrid), len(decgrid), 3))

    for i in range(len(z_list)):
        for j in range(len(ragrid)):
            for k in range(len(decgrid)):

                cube_voxel_address[i,j,k,0] = Dc_center[i]
                cube_voxel_address[i,j,k,1] = ys[j, i]
                cube_voxel_address[i,j,k,2] = xs[k, i]

    dc_center = cube_voxel_address[int(cube_voxel_address.shape[0]/2),int(cube_voxel_address.shape[1]/2),int(cube_voxel_address.shape[2]/2),0]
    centered_cube = cube_voxel_address.copy()
    centered_cube[:,:,:,0] -= dc_center
    dict = {'cube_mpc_centered':centered_cube}
    pickle.dump(dict, open(pars['output_path']+f'{cat_name}_cube_3D_z{z_center}_MJy_sr_{line}.p', 'wb'))
    '''

if __name__ == "__main__":

    params = load_params('PAR_FILES/SIDES_from_original_with_fir_lines.par')
    TIM_params = load_params('PAR_FILES/Uchuu_cubes_for_TIM.par')

    for tile_sizeRA, tile_sizeDEC in TIM_params['tile_sizes']: 
        print(tile_sizeRA, tile_sizeDEC)
        
        # List files matching the pattern
        files = sorted_files_by_n(TIM_params["sides_cat_path"], ((tile_sizeRA, tile_sizeDEC),))
        
        bar = Bar('Processing', max=len(files))

        for l, file in enumerate(files):

            cat = Table.read(TIM_params["sides_cat_path"]+file)
            cat = cat.to_pandas()
    
            for z_center, dz in zip(TIM_params['z_centers'], TIM_params['dz']): 

                gen_comoving_cube(cat, file[:-5], TIM_params, 'CII_de_Looze', freq_CII,
                                  z_center=z_center, Delta_z=dz)

            bar.next()
        bar.finish

import sys
import os
from analysis_fcts.fcts import * 
from pysides.make_cube import *
from pysides.load_params import *
import argparse
import time
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import glob
import sys

def list_files(pattern):
    files = glob.glob(pattern)
    return files

freq_CII = 1900.53690000 * u.GHz
freq_CI10 = 492.16 *u.GHz
freq_CI21 = 809.34 * u.GHz
rest_freq_list = [115.27120180  *u.GHz* J_up for J_up in range(1, 9)]
rest_freq_list.append(freq_CI10); rest_freq_list.append(freq_CI21); rest_freq_list.append(freq_CII); 
line_list = ["CO{}{}".format(J_up, J_up - 1) for J_up in range(1, 9)]
line_list.append('CI10'); line_list.append('CI21'); line_list.append('CII_de_Looze')

def p_of_k_for_comoving_cube(cube_file_name, TIM_params):

    if(not '3D' in cube_file_name): print('warning !')

    cube = fits.getdata(cube_file_name)
    hdr = fits.getheader(cube_file_name)

    normpk = hdr['CDELT1'] * hdr['CDELT2'] *hdr['CDELT3'] / (hdr['NAXIS1'] * hdr['NAXIS2'] * hdr['NAXIS3'])
    pow_sqr = np.absolute(np.fft.fftn(cube)**2 * normpk )

    w_freq = 2*np.pi*np.fft.fftfreq(hdr['NAXIS1'], d=hdr['CDELT1'])
    v_freq = 2*np.pi*np.fft.fftfreq(hdr['NAXIS2'], d=hdr['CDELT2'])
    u_freq = 2*np.pi*np.fft.fftfreq(hdr['NAXIS3'], d=hdr['CDELT3'])

    k_sphere_freq = np.sqrt(u_freq[:,np.newaxis,np.newaxis]**2 + v_freq[np.newaxis,:,np.newaxis]**2 + w_freq[np.newaxis,np.newaxis,:]**2)/ u.Mpc
    k_transv_freq = np.sqrt( v_freq[:,np.newaxis,np.newaxis]**2 + w_freq[np.newaxis,:,np.newaxis]**2) / u.Mpc
    k_transv_freq_3d = np.zeros(k_sphere_freq.shape)
    k_transv_freq_3d[:,:,:] = k_transv_freq[:,:,0][np.newaxis,:,:]   
    k_z_freq =      np.sqrt( u_freq**2 ) / u.Mpc    
    k_z_freq_3d = np.zeros(k_sphere_freq.shape)
    k_z_freq_3d[:,:,:] = k_z_freq[:,np.newaxis, np.newaxis]  
    k_cylindrical = np.zeros((hdr['NAXIS3'], hdr['NAXIS2'], hdr['NAXIS1'],2))
    k_cylindrical[:,:,:,0] = k_z_freq_3d
    k_cylindrical[:,:,:,1] = k_transv_freq_3d

    delta_k_over_k = TIM_params['dkk']

    #k_nyquist = 1 / 2 / res.to(u.rad)  #rad**-1
    k_bintab_sphere, k_binwidth_sphere = make_bintab((k_sphere_freq.min(),k_sphere_freq.max()), 0.1/ u.Mpc, delta_k_over_k) 
    k_bintab_transv, k_binwidth_transv = make_bintab((k_transv_freq.min(),k_transv_freq.max()), 0.1/ u.Mpc, delta_k_over_k) 
    k_bintab_z, k_binwidth_z           = make_bintab((k_z_freq.min(),k_z_freq.max()), 0.1/ u.Mpc, delta_k_over_k) 

    k_out_sphere, edges = np.histogram(k_sphere_freq, bins = k_bintab_sphere, weights = k_sphere_freq)
    pk_out_sphere, edges = np.histogram(k_sphere_freq, bins = k_bintab_sphere, weights = pow_sqr)
    histo, edegs = np.histogram(k_sphere_freq, bins = k_bintab_sphere)
    k_out_sphere /= histo
    pk_out_sphere /= histo

    '''
    k_out_transv, edges = np.histogramdd(k_transv_freq_3d, bins = k_bintab_transv, weights = k_transv_freq_3d)
    pk_out_transv, edges = np.histogram(k_transv_freq_3d, bins = k_bintab_transv, weights = pow_sqr)
    histo, edegs = np.histogram(k_transv_freq_3d, bins = k_bintab_transv)
    k_out_transv /= histo
    pk_out_transv /= histo

    k_out_z, edges = np.histogram(k_z_freq, bins = k_bintab_z, weights = k_z_freq)
    pk_out_z, edges = np.histogram(k_z_freq, bins = k_bintab_z, weights = pow_sqr)
    histo, edegs = np.histogram(k_z_freq, bins = k_bintab_z)
    k_out_z /= histo
    pk_out_z /= histo
    '''

    histo, edges = np.histogramdd((k_z_freq_3d.ravel(), k_transv_freq_3d.ravel()), 
                                  bins=(k_bintab_z.value, k_bintab_transv.value))
    
    # Compute the weighted sums for k_z and k_transv
    k_out_z = np.histogramdd((k_z_freq_3d.ravel(), k_transv_freq_3d.ravel()), 
                             bins=(k_bintab_z.value, k_bintab_transv.value), 
                             weights=k_z_freq_3d.ravel())[0]
    
    k_out_transv = np.histogramdd((k_z_freq_3d.ravel(), k_transv_freq_3d.ravel()), 
                                  bins=(k_bintab_z.value, k_bintab_transv.value), 
                                  weights=k_transv_freq_3d.ravel())[0]

    pk_out = np.histogramdd((k_z_freq_3d.ravel(), k_transv_freq_3d.ravel()), 
                            bins=(k_bintab_z.value, k_bintab_transv.value), 
                            weights=pow_sqr.ravel())[0]

    # Normalize by the histogram counts
    k_out_z /= histo
    k_out_transv /= histo
    pk_out /= histo

    # Set up the figure and axis
    fig, (axsphere, axcyl) = plt.subplots(1,2,figsize=(8, 8))
    axsphere.loglog(k_out_sphere, pk_out_sphere, 'k')
    axsphere.set_title('Spherical power spectrum')
    axsphere.set_ylabel('Power Spectrum $\\rm P(k) [Jy^2/sr^2.Mpc^3]$')
    axsphere.set_xlabel('$\\rm k$ [$\\rm Mpc^{-1}$]')
    # Use pcolormesh to create the 2D histogram plot with logarithmic color scaling
    # We need to provide the bin edges for the plot
    k_z_edges, k_transv_edges = edges
    axcyl.set_title('Cylindrical power spectrum')
    c = axcyl.pcolormesh(k_z_edges, k_transv_edges, pk_out.T, 
                      shading='auto', cmap='viridis', norm=LogNorm())
    # Add a colorbar
    colorbar = plt.colorbar(c, ax=axcyl)
    colorbar.set_label('Power Spectrum $\rm P(k) [Jy^2/sr^2.Mpc^3]$')
    # Set the axis labels
    axcyl.set_xlabel('$\\rm k_{\\parallel}$ [$\\rm Mpc^{-1}$]')
    axcyl.set_ylabel('$\\rm k_{\\perp}$ [$\\rm Mpc^{-1}$]')
    # Set log scales for the axes
    axcyl.set_xscale('log')
    axcyl.set_yscale('log')
    # Show the plot
    fig.tight_layout()
    plt.show()

    embed()

    return k_out_sphere, k_out_transv, k_out_z, pk_out_sphere, pk_out

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

def gen_comoving_cube(cat, cat_name, pars, line, rest_freq, z_center=6, Delta_z=0.5):

    cat = cat.loc[np.abs(cat['redshift']-z_center) <= Delta_z/2]

    nu_obs = rest_freq / (1+z_center)
    dz = ( pars['freq_resol'] / 1e9 ) * (1+z_center) / nu_obs.value
    z_bins = np.arange(z_center-Delta_z/2-dz/2, z_center+Delta_z/2+dz/2, dz)
    z_list = np.arange(z_center-Delta_z/2, z_center+Delta_z/2, dz)

    res = (pars['pixel_size'] *u.arcsec).to(u.deg).value

    ragrid_bins =np.arange(cat['ra'].min() -res/2,cat['ra'].max() +res/2,res)
    decgrid_bins=np.arange(cat['dec'].min()-res/2,cat['dec'].max()+res/2,res)

    angular_grid =  np.array(np.meshgrid(ragrid_bins,decgrid_bins))

    embed()

    cube_MJy_per_sr_per_Mpc, edges = np.histogramdd(sample = (np.asarray(cat['redshift']), cat['ra'], cat['dec']), 
                                            bins = (z_bins, ragrid_bins, decgrid_bins), weights = cat[f'I{line}'])

    #compute the coordinates in the cube in comoving Mpc
    Dc_center = cosmo.comoving_distance(z_list).value
    ragrid =np.arange(cat['ra'].min() , cat['ra'].max(),res)
    decgrid=np.arange(cat['dec'].min(),cat['dec'].max(),res)
    ra_center = np.mean(ragrid)
    dec_center = np.mean(decgrid)
    ys = Dc_center * ( ragrid[:, np.newaxis]  - ra_center)  * (np.pi/180) * np.cos(np.pi/180*ragrid[:, np.newaxis])
    xs = Dc_center * ( decgrid[:, np.newaxis] - dec_center) * (np.pi/180)

    #convert to the proper unit (MJy/sr/Mpc3)
    transverse_res_list = []
    radial_res_list = []
    for i, z in enumerate(z_list):
        dv_voxel = np.abs(ys[0,i]-ys[1,i]) * np.abs(xs[0,i]-xs[1,i]) * (cosmo.comoving_distance(z+dz/2) - cosmo.comoving_distance(z-dz/2)).value
        cube_MJy_per_sr_per_Mpc[i,:,:] *=  1.e-6 / dv_voxel
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

    output_name = f'{pars["output_path"]}/{cat_name}_cube_3D_z{z_center}_MJy_sr_{line}.fits'
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

    embed()

if __name__ == "__main__":

    params = load_params('PAR_FILES/SIDES_from_original_with_fir_lines.par')
    TIM_params = load_params('PAR_FILES/Uchuu_cubes_for_TIM.par')

    #With SIDES Bolshoi, for rapid tests. 
    '''
    dirpath="/home/mvancuyck/"
    cat = Table.read(dirpath+'pySIDES_from_original.fits')
    cat = cat.to_pandas()
    simu='pySIDES_from_bolshoi'; fs=2
    '''

    for tile_sizeRA, tile_sizeDEC in TIM_params['tile_sizes']: 

        pattern = f'{TIM_params["output_path"]}pySIDES_from_uchuu_tile_*_{tile_sizeRA}deg_x_{tile_sizeDEC}deg.fits'
        # List files matching the pattern
        files = list_files(pattern)
        
        for file in files:

            cat = Table.read(file)
            cat = cat.to_pandas()
    
            gen_spatial_spectral_cube(cat, file[len(TIM_params["output_path"]):-5], TIM_params)

            for z_center, dz in zip(TIM_params['z_centers'], TIM_params['dz']): 

                gen_comoving_cube(cat, file[len(TIM_params["output_path"]):-5], TIM_params, 'CII_de_Looze', freq_CII,
                                  z_center=z_center, Delta_z=dz)

                pk_cylindre, k_2d = p_of_k_for_comoving_cube(f'{TIM_params["output_path"]}/{file[len(TIM_params["output_path"]):-5]}_cube_3D_z{z_center}_MJy_sr_CII_de_Looze.fits', TIM_params)
                #pk_cylindre, k_2d = pk_spherical()



    """
def powspec(pos_x,pos_y,pos_z,
            l_line,redshift,
            cosmo,box_edge_no_h,wavelength,kbins):
    
    if len(pos_x) == 0:
        return None

    d = cosmo.comoving_distance(redshift).value       # comoing distance at z in Mpc
    dl = (1+redshift)*d                               # luminosity distance to z in Mpc
    y = wavelength * (1+redshift)**2 / (1000*cosmo.H(redshift).value)  # derivative of distance with respect to frequency in Mpc / Hz

    pixel_size = 1
    side_length = (box_edge_no_h//pixel_size)*pixel_size   # Cut off the edge of the box if it doesn't match pixel size
    center_point = np.array([side_length/2,side_length/2,side_length/2])

    positions = np.array([pos_x,pos_y,pos_z]).T

    intensities = l_line / pixel_size**3 / (4*np.pi*dl**2) * d**2 * y  # in Lsun/Mpc^3 / Mpc^2 * Mpc^2/Sr * Mpc/Hz
    intensities = intensities * 3.828e26 / 3.0857e22**2 *1e26               # in Jy/Sr
    intensities[np.isnan(intensities)] = 0

    grid = Gridder(positions,intensities,center_point=center_point,side_length=side_length,pixel_size=pixel_size,axunits='Mpc',gridunits='Jy/Sr')
    ps = grid.power_spectrum(in_place=False,normalize=True)

    ax1d, ps1d = ps.spherical_average(ax=[0,1,2],shells=kbins/(2*np.pi))
    ps1d = ps1d[:,0] / side_length**3

    return ps1d
    """
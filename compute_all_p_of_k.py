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
import matplotlib.patches as patches
from progress.bar import Bar
from gen_all_sizes_TIM_cubes import sorted_files_by_n

import glob
import sys

def list_files(pattern):
    files = glob.glob(pattern)
    return files

freq_CII = 1900.53690000 * u.GHz
freq_CI21 = 809.34 * u.GHz
rest_freq_list = [115.27120180  *u.GHz* J_up for J_up in range(7, 9)]
rest_freq_list.append(freq_CI21); rest_freq_list.append(freq_CII); 
line_list = ["CO{}{}".format(J_up, J_up - 1) for J_up in range(7, 9)]
line_list.append('CI21'); line_list.append('CII_de_Looze')

def p_of_k_for_comoving_cube(cat_name,line,rest_freq, z_center, Delta_z,  filecat, pars, recompute=False):

    path = pars['output_path']
    dict_pks_name = f'dict_dir/{cat_name}_cube_3D_z{z_center}_Jy_sr_{line}_pk3d.p'
    dico_exists = os.path.isfile(dict_pks_name)
    key_exists = False
    if(dico_exists): 
        dico_loaded = pickle.load( open(dict_pks_name, 'rb'))
        key_exists = ('shot noise #Jy/sr' in dico_loaded.keys())
    #--- 
    if(not key_exists or recompute):

        cube = fits.getdata(pars['output_path']+'/'+f'{cat_name}_cube_3D_z{z_center}_Jy_sr_{line}.fits')
        gal  = fits.getdata(pars['output_path']+'/'+f'{cat_name}_cube_3D_z{z_center}_galaxies.fits')
        gal /= gal.mean()
        gal -= 1
        hdr = fits.getheader(pars['output_path']+'/'+f'{cat_name}_cube_3D_z{z_center}_Jy_sr_{line}.fits')

        normpk = hdr['CDELT1'] * hdr['CDELT2'] *hdr['CDELT3'] / (hdr['NAXIS1'] * hdr['NAXIS2'] * hdr['NAXIS3'])
        pow_sqr = np.absolute(np.fft.fftn(cube)**2 * normpk )
        pow_cross = np.real(np.fft.fftn(cube)* np.conj(np.fft.fftn(gal))* normpk )

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

        delta_k_over_k = TIM_params['dkk']

        #k_nyquist = 1 / 2 / res.to(u.rad)  #rad**-1
        k_bintab_sphere, k_binwidth_sphere = make_bintab((k_sphere_freq[k_sphere_freq.value>0].min().value,k_sphere_freq.max().value), 0.01, delta_k_over_k) 
        k_bintab_transv, k_binwidth_transv = make_bintab((k_transv_freq[k_transv_freq.value>0].min().value,k_transv_freq.max().value), 0.3, delta_k_over_k) 
        k_bintab_z, k_binwidth_z           = make_bintab((k_z_freq[k_z_freq.value>0].min().value,k_z_freq.max().value), 0.01/ u.Mpc, delta_k_over_k) 

        k_out_z, e = np.histogram(k_z_freq_3d, bins = k_bintab_z.value, weights = k_z_freq_3d)
        histo_z, e = np.histogram(k_z_freq_3d, bins = k_bintab_z.value)
        k_out_z /= histo_z

        k_out_transv, e = np.histogram(k_transv_freq_3d, bins = k_bintab_transv.value, weights = k_transv_freq_3d)
        histo_transv, e = np.histogram(k_transv_freq_3d, bins = k_bintab_transv.value)
        k_out_transv /= histo_transv

        k_out_sphere, e = np.histogram(k_sphere_freq, bins = k_bintab_sphere, weights = k_sphere_freq)
        pk_out_sphere, e = np.histogram(k_sphere_freq, bins = k_bintab_sphere, weights = pow_sqr)
        xpk_out_sphere, e = np.histogram(k_sphere_freq, bins = k_bintab_sphere, weights = pow_cross)
        histo_sphere, e = np.histogram(k_sphere_freq, bins = k_bintab_sphere)
        k_out_sphere /= histo_sphere
        pk_out_sphere /= histo_sphere
        xpk_out_sphere /= histo_sphere

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
        xpk_out = np.histogramdd((k_z_freq_3d.ravel(), k_transv_freq_3d.ravel()), 
                        bins=(k_bintab_z.value, k_bintab_transv.value), 
                        weights=pow_cross.ravel())[0]

        # Normalize by the histogram counts
        k_out_z /= histo
        k_out_transv /= histo
        pk_out /= histo
        xpk_out /= histo
        '''
        # Set up the figure and axis
        fig, axs = plt.subplots(2,2,figsize=(8,4), sharex=True, dpi=200)
        axsphere, axcyl = axs[0,0], axs[0,1]
        axspherelow = axs[1,0]
        axspherecross = axs[1, 1]
        # Use pcolormesh to create the 2D histogram plot with logarithmic color scaling
        # We need to provide the bin edges for the plot
        k_z_edges, k_transv_edges = edges
        axcyl.set_title('Cylindrical power spectrum')
        c = axcyl.pcolormesh(k_bintab_transv, k_bintab_z,  pk_out, 
                            shading='auto', cmap='viridis', norm=LogNorm()) #vmin = pk_out_sphere.min(), vmax=pk_out_sphere.max() 
        # Add a colorbar
        colorbar = plt.colorbar(c, ax=axcyl)
        colorbar.set_label('$\\rm P(k)$ $\\rm[Jy^2/sr^2.Mpc^3]$')
        # Set the axis labels
        axcyl.set_ylabel('$\\rm k_{\\parallel}$ [$\\rm Mpc^{-1}$]')
        axcyl.set_xlabel('$\\rm k_{\\perp}$ [$\\rm Mpc^{-1}$]')
        # Set log scales for the axes
        axcyl.set_ylim(k_bintab_z.min().value,k_bintab_z.max().value)
        axcyl.set_xlim(1e-3,2e1)#k_bintab_transv.min().value,k_bintab_transv.max().value)

        for k in k_out_sphere:
            circle = patches.Circle((0., 0.), k.value, edgecolor='r', facecolor='none', alpha=0.1)
            axcyl.add_patch(circle)
        axcyl.set_xscale('log')
        axcyl.set_yscale('log')
        axsphere.loglog(k_out_sphere, pk_out_sphere, '-ok', markersize=1.5)
        axsphere.set_title('Spherical power spectrum')
        axsphere.set_ylabel('$\\rm P(k)$ $\\rm[Jy^2/sr^2.Mpc^3]$')
        axspherecross.loglog(k_out_sphere, xpk_out_sphere, '-ok', markersize=1.5)
        axspherecross.set_ylabel('$\\rm P_X(k)$ $\\rm[Jy/sr.Mpc^3]$')
        axspherecross.set_xlabel('$\\rm k$ [$\\rm Mpc^{-1}$]')

        axspherelow.set_xlabel('$\\rm k$ [$\\rm Mpc^{-1}$]')
        axspherelow.set_ylabel('Nb count of modes')
        axspherelow.stairs(histo_sphere,k_bintab_sphere.value, color='r', label='spherical modes')
        axspherelow.stairs(histo_transv,k_bintab_transv.value, color='g', label='$\\rm \\perp$ modes')
        axspherelow.stairs(histo_z,k_bintab_z.value, color='b', label='$\\rm \\parallel$ modes')
        axspherelow.set_yscale('log')
        axspherelow.legend(bbox_to_anchor=(1,0.8), frameon=False)
        # Show the plot
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        plt.rcParams.update({'font.size': 10})
        plt.rcParams.update({'xtick.direction':'in'})
        plt.rcParams.update({'ytick.direction':'in'})
        plt.rcParams.update({'xtick.top':True})
        plt.rcParams.update({'ytick.right':True})
        plt.rcParams.update({'legend.frameon':False})

        fig.savefig(f'figures/{cat_name}_cube_3D_z{z_center}_Jy_sr_{line}_3dpk_tab.png', transparent=True)

        plt.close()
        '''

        dict = {'k_out_sphere #Mpc-1':k_out_sphere, 
                'k_out_transv #Mpc-1':k_out_transv, 
                'k_out_z #Mpc-1':k_out_z, 
                'k_bins #Mpc-1':edges,
                'k_bins_sphere #Mpc-1':k_bintab_sphere,
                'pk_out_sphere #Jy2sr-2Mpc3':pk_out_sphere, 
                'pk_out #Jy2sr-2Mpc3':pk_out,
                'cross pk_out_sphere #Jysr-1Mpc3':xpk_out_sphere,
                'cross pk_out #Jysr-1Mpc3':xpk_out,
                'nb_count_sphere':histo_sphere,
                'nb_count_transv':histo_transv,
                'nb_count_z':histo_z}

        cat = Table.read(f'{path}/'+filecat)
        cat = cat.to_pandas()
        cat = cat.loc[np.abs(cat['redshift']-z_center) <= Delta_z/2]
        L = cat[f'I{line}'] * 1.04e-3 * cat['Dlum']**2 * rest_freq/(1+cat['redshift'])
        I = L * (cst.c*1e-3) * 4.02e7 / (4*np.pi) / rest_freq.to(u.Hz).value / cosmo.H(cat['redshift']).value
        #------
        Vcube = (hdr['CDELT1'] * hdr['CDELT2'] *hdr['CDELT3']) * (hdr['NAXIS1'] * hdr['NAXIS2'] * hdr['NAXIS3'])
        dict['I #Jy/sr']= np.sum(I) / Vcube
        dict['shot noise #Jy/sr']= np.sum(I**2) / Vcube

        if(not dico_exists): 
            print("save the dict "+dict_pks_name)
            pickle.dump(dict, open(dict_pks_name, 'wb'))
        else: 
            print("update the dict"+dict_pks_name)
            dico_loaded.update(dict)
            pickle.dump(dico_loaded, open(dict_pks_name, 'wb'))

    else: print('load the dict'+dict_pks_name)

    dict = pickle.load( open(dict_pks_name, 'rb'))

    return dict


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

def naive_NU_DF_PK_for_angular_spectral_cube(z_center, Delta_z, cubefile='OUTPUT_TIM_CUBES_FROM_UCHUU/pySIDES_from_uchuu_TIM_tile99_0.2deg_1deg_CII_de_Looze_nobeam_MJy_sr.fits',
                                        rest_freq = 1900.53690000 * u.GHz, dkk=0.1):

    #Naive Non-Uniform power spectrum
    #I need to investigate more references.

    #Load the angular spectral cube and its header 
    hdu = fits.open(cubefile)
    hdr = hdu[0].header
    angular_cube = (hdu[0].data * u.MJy/u.sr).to(u.Jy/u.sr) #u.Unit(hdr['BUNIT'])
    w = wcs.WCS(hdr)   
    
    #Set the redshift axis  append=x[-1]
    freqs = np.linspace(0,hdr['NAXIS3'],hdr['NAXIS3']+1)
    freq_list = w.swapaxes(0, 2).sub(1).wcs_pix2world(freqs, 0)[0] * u.Unit(hdr['CUNIT3'])

    freqmin = (rest_freq.to(hdr['CUNIT3'])) / (1+z_center+Delta_z/2)
    freqmax = (rest_freq.to(hdr['CUNIT3'])) / (1+z_center-Delta_z/2)
    fmin = np.where(freq_list>= freqmin)[0][0]
    fmax = np.where(freq_list<= freqmax)[0][-1]
    freq_list = freq_list[fmin:fmax+2]
    angular_cube = angular_cube[fmin:fmax+1]

    redshift_list = (rest_freq.to(hdr['CUNIT3']) / freq_list - 1 ).value    
    #xpixlist = np.linspace(0,hdr['NAXIS2'],hdr['NAXIS2']+1)
    #ypixlist = np.linspace(0,hdr['NAXIS1'],hdr['NAXIS1']+1)
    Nmax = np.max((hdr['NAXIS2'],hdr['NAXIS1']))

    xpixlist = np.linspace(0,Nmax,Nmax+1)
    ypixlist = np.linspace(0,Nmax,Nmax+1)
    ra ,dec = w.celestial.wcs_pix2world( ypixlist, xpixlist, 0)*u.deg
    xpixgrid = list(np.linspace(-0.5,hdr['NAXIS2']+0.5,hdr['NAXIS2']+2))
    ypixgrid = list(np.linspace(-0.5,hdr['NAXIS1']+0.5,hdr['NAXIS1']+2))
    ragrid , decgrid = w.celestial.wcs_pix2world( ypixlist, xpixlist , 0) *u.deg
    ra_center = np.mean(ragrid)
    dec_center = np.mean(decgrid)
    delta_ra = np.max(ragrid) - np.min(ragrid)
    delta_dec = np.max(decgrid) - np.min(decgrid)
    #all the coordinates will be in comoving units
    Dc_center = cosmo.comoving_distance(redshift_list)

    #compute the coordinates in the cube
    y_Mpc = Dc_center[:,np.newaxis] * (( ra[ np.newaxis,:]  - ra_center)  * (np.pi/180) * np.cos(np.pi/180*ragrid[ np.newaxis,:])).value
    x_Mpc = Dc_center[:,np.newaxis] * (( dec[ np.newaxis,:] - dec_center) * (np.pi/180)).value

    dept_vox = np.abs(np.diff(Dc_center))
    area_vox = np.zeros(( int(fmax-fmin+1), Nmax , Nmax ))  
    area_vox[:,:,:] = np.diff(y_Mpc[:1,:], axis=1)[:,:,np.newaxis] * np.diff(x_Mpc[:1,:], axis=1)[:,np.newaxis,:]
    v_vox = dept_vox[:,np.newaxis, np.newaxis] * area_vox[:, :,:]   
    angular_cube /= v_vox[:,:hdr['NAXIS2'],:hdr['NAXIS1']]


    res_perp = (area_vox[:,:hdr['NAXIS2'],:hdr['NAXIS1']].mean())**(1/2)
    res_rad  =  dept_vox.mean()
    res_avg = (v_vox[:,:hdr['NAXIS2'],:hdr['NAXIS1']].mean())*(1/3)

    kz = 2*np.pi*give_map_spatial_freq_one_axis(res_rad, len(Dc_center))
    kmaps_list = []
    for z in range(int(fmax-fmin+1)):
        res = (area_vox[z,:hdr['NAXIS2'],:hdr['NAXIS1']].mean())
        kmap = 2*np.pi*give_map_spatial_freq(res, angular_cube.shape[1], angular_cube.shape[2])
        kmaps_list.append(kmap)

    kmaps_list = np.asarray(kmaps_list)
    k = np.sqrt(np.asarray(kmaps_list)**2+kz[:-1,np.newaxis, np.newaxis].value**2)/u.Mpc
    dk_min = 2 / ( np.min([hdr['NAXIS1'],hdr['NAXIS2'],hdr['NAXIS3']]) * res_avg )

    k_bins, k_width = make_bintab(k.value, dk_min.value, dkk=dkk)

    normpk =  v_vox[:,:hdr['NAXIS2'],:hdr['NAXIS1']] / (hdr['NAXIS1'] * hdr['NAXIS2'] * hdr['NAXIS3'])
    pow_sqr = np.absolute(np.fft.fftn(angular_cube)**2 * normpk )

    norm_k, k_edges = np.histogram(k.value, bins=k_bins)
    k_adress, k_edges = np.histogram(k.value, bins=k_bins, weights=k)
    k_adress /= norm_k
    p_of_k, k_edges = np.histogram(k.value, bins=k_bins, weights=pow_sqr)
    p_of_k /= norm_k

    plt.loglog(k_adress, p_of_k)
    plt.show()

    return k_adress, p_of_k

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

        # List files matching the pattern
        files = sorted_files_by_n(TIM_params["output_path"], ((tile_sizeRA, tile_sizeDEC),))
        dict_fieldsize = {}

        for l, file in enumerate(files):
            
            dictl = {}

            for z_center, dz in zip(TIM_params['z_centers'], TIM_params['dz']): 

                dictl[f'pk_3D_z{z_center}_CII_de_Looze'] = p_of_k_for_comoving_cube(file[:-5],'CII_de_Looze',freq_CII, z_center, dz, file, TIM_params)

                dictl[f'pk_3D_non-uniform_z{z_center}_CII_de_Looze'] = naive_NU_DF_PK_for_angular_spectral_cube(z_center, dz,
                                                                                                                cubefile=f'OUTPUT_TIM_CUBES_FROM_UCHUU/pySIDES_from_uchuu_TIM_tile{l}_{tile_sizeRA}deg_{tile_sizeDEC}deg_CII_de_Looze_nobeam_MJy_sr.fits',)

            dict_fieldsize[f'{l}'] = dictl

        pickle.dump(dict_fieldsize, open('dict_dir/'+f'pySIDES_from_uchuu_{tile_sizeRA}deg_x_{tile_sizeDEC}deg_pks.p', 'wb'))

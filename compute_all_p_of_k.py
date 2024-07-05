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

def p_of_k_for_comoving_cube(cat_name,line,z_center, pars, recompute=False):

    path = pars['output_path']
    dict_pks_name = f'dict_dir/{cat_name}_cube_3D_z{z_center}_Jy_sr_{line}_pk3d.p'
    dico_exists = os.path.isfile(dict_pks_name)
    key_exists = False
    if(dico_exists): 
        dico_loaded = pickle.load( open(dict_pks_name, 'rb'))
        key_exists = ('nb_count_sphere' in dico_loaded.keys())
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
        k_bintab_sphere, k_binwidth_sphere = make_bintab((k_sphere_freq[k_sphere_freq.value>0].min(),k_sphere_freq.max()), 0.01/ u.Mpc, delta_k_over_k) 
        k_bintab_transv, k_binwidth_transv = make_bintab((k_transv_freq[k_transv_freq.value>0].min(),k_transv_freq.max()), 0.3/ u.Mpc, delta_k_over_k) 
        k_bintab_z, k_binwidth_z           = make_bintab((k_z_freq[k_z_freq.value>0].min(),k_z_freq.max()), 0.01/ u.Mpc, delta_k_over_k) 

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
                'pk_out_sphere #Jy2sr-2Mpc3':pk_out_sphere, 
                'pk_out #Jy2sr-2Mpc3':pk_out,
                'cross pk_out_sphere #Jysr-1Mpc3':xpk_out_sphere,
                'cross pk_out #Jysr-1Mpc3':xpk_out,
                'nb_count_sphere':histo_sphere,
                'nb_count_transv':histo_transv,
                'nb_count_z':histo_z}

        #pickle.dump(dict, open(f'{cube_file_name}_3dpk.p', 'wb'))

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

def angular_p_of_k(catname, pars, line, rest_freq=None, 
                   recompute=False, to_emb=False):

    #----
    path = pars['output_path']
    dict_pks_name = f'dict_dir/{catname}_{line}.p'
    dico_exists = os.path.isfile(dict_pks_name)
    key_exists = False
    if(dico_exists): 
        dico_loaded = pickle.load( open(dict_pks_name, 'rb'))
        key_exists = ('k #/arcmin' in dico_loaded.keys())
    #--- 
    if(not key_exists or recompute):

        if(to_emb): embed()

        dkk = pars['dkk']

        LIM = fits.getdata(f"{path}/{catname}_{line}_MJy_sr.fits")*u.MJy/u.sr
        LIM_smoothed = fits.getdata(f"{path}/{catname}_{line}_MJy_sr_smoothed.fits")*u.MJy/u.sr
        Gal = fits.getdata(f"{path}/{catname}_galaxies_pix.fits")

        hdr = fits.getheader(f"{path}/{catname}_{line}_MJy_sr.fits")
        res = hdr["CDELT1"]* u.Unit(hdr["CUNIT1"]) 
        npix = hdr["NAXIS1"] * hdr["NAXIS2"]
        field_size = ( npix * res**2 ).to(u.deg**2)
        dnu = np.round( (hdr["CDELT3"] * u.Unit(hdr["CUNIT3"])).to(u.GHz).value, 2)
        w = wcs.WCS(hdr)  
        spec_axis = np.arange(0,hdr['NAXIS3'],1)
        freq_list = w.swapaxes(0, 2).sub(1).wcs_pix2world(spec_axis, 0)[0] * u.Unit(hdr["CUNIT3"])

        k_nyquist, k_min, delta_k, k_bintab, k, k_map, nb_mode_count = set_k_infos(hdr["NAXIS2"],hdr["NAXIS1"], res, delta_k_over_k = dkk)

        pk_im_1d = []
        pk_imB_1d = []
        pk_p2 = []
        pk_im_1d_smoothed = []
        pk_p2_smoothed = []
        for i in range(LIM.shape[0]):
            #----x
            gal_mean =  Gal[i,:,:].mean()
            if(gal_mean <0): continue
            else: Gal[i,:,:] /= gal_mean
            #----            
            pk, _ = my_p2(LIM[i,:,:], res, k_bintab, u.Jy**2/u.sr, u.arcmin**-1 )
            pk_im_1d.append( pk.value ) 
            p_imB, _ = my_p2(Gal[i,:,:], res, k_bintab, u.sr, u.arcmin**-1 )
            pk_imB_1d.append(p_imB.value)
            pk, _    = my_p2(Gal[i,:,:], res, k_bintab, u.Jy, u.arcmin**-1, map2 = LIM[i,:,:])
            pk_p2.append(pk.value)

            pk, _ = my_p2(LIM_smoothed[i,:,:], res, k_bintab, u.Jy**2/u.sr, u.arcmin**-1 )
            pk_im_1d_smoothed.append( pk.value ) 
            pk, _    = my_p2(Gal[i,:,:], res, k_bintab, u.Jy, u.arcmin**-1, map2 = LIM_smoothed[i,:,:])
            pk_p2_smoothed.append(pk.value)

        dict = {'species':line,
                'cube':f"{path}/{catname}_{line}_MJy_sr.fits", 
                'Galaxies':f"{path}/{catname}_galaxies_pix.fits",
                'dnu #GHz':dnu,
                'k #/arcmin': k.to(u.arcmin**-1),
                'k_map #/arcmin':k_map.to(u.arcmin**-1),
                'k_nyquist #/arcmin':  k_nyquist.to(u.arcmin**-1),
                'k_min #/arcmin':k_min.to(u.arcmin**-1),
                'kbin #/arcmin':k_bintab.to(u.arcmin**-1),
                'nb_mode_count': nb_mode_count,
                'res #rad':res.to(u.rad),
                'pk_species #Jy2/sr': np.asarray(pk_im_1d),
                'pk_gal #sr': np.asarray(pk_imB_1d),      
                'pk_species-gal #Jy': np.asarray(pk_p2),
                'pk_species smoothed #Jy2/sr': np.asarray(pk_im_1d_smoothed),
                'pk_species-gal smoothed #Jy': np.asarray(pk_p2_smoothed),
                'freq_list #Hz':freq_list}
        # Set up the figure and axis
        fig, (axJ, axG, axJG,axnb) = plt.subplots(1,4,figsize=(16,4), dpi=200)
        for i, (ax, y,y_smoothed, unit) in enumerate(zip((axJ, axG, axJG),
                                            (dict['pk_species #Jy2/sr'],dict['pk_gal #sr'],dict['pk_species-gal #Jy']),
                                            (dict['pk_species smoothed #Jy2/sr'],dict['pk_gal #sr'],dict['pk_species-gal smoothed #Jy']),
                                            ('$\\rm P(k) [Jy^2/sr$]', '$\\rm P_G(k) [sr]$', '$\\rm P_X(k) [Jy/sr]$'))):
            x = dict['k #/arcmin']
            if(i==1): ax.set_title(f"{catname}_{line}, \n res={np.round(dict['res #rad'].to(u.arcsec), 2)}")
            for i,c in zip(range(LIM.shape[0]), cm.plasma(np.linspace(0,1,LIM.shape[0]))):
                ax.loglog(x, y[i,:], c=c)
                ax.loglog(x, y_smoothed[i,:], c=c, ls=':')
            ax.set_xlabel('k [$\\rm arcmin^{-1}$]')
            ax.set_ylabel(unit)
        axnb.loglog(x, dict['nb_mode_count'], '-ok')
        axnb.set_xlabel('k [$\\rm arcmin^{-1}$]')
        axnb.set_ylabel('Nb count of modes')
        fig.tight_layout()
        fig.savefig('figures/{catname}_{line}.png', transparent=True)
        plt.close()
        
        if(rest_freq is not None):

            z_list = (rest_freq.to(u.GHz) / freq_list.to(u.GHz) ).value - 1 

            cat = Table.read(f'{path}/'+file)
            cat = cat.to_pandas()

            spec_axis_edges = np.arange(0-0.5,hdr['NAXIS3']+0.5,1)
            freqs_edges = w.swapaxes(0, 2).sub(1).wcs_pix2world(spec_axis_edges, 0)[0] * u.Unit(hdr["CUNIT3"])
            #------
            freq_obs = (rest_freq.value / (1 + cat['redshift']))
            vdelt = (cst.c * 1e-3) * dnu / freq_obs #km/s
            #------
            hist, edges = np.histogram( freq_obs, bins = freqs_edges.to(u.GHz).value, weights = (cat[f"I{line}"]/vdelt)**2)
            p_snlist = hist * u.Jy**2 / field_size.to(u.sr) 
            dict['LIM_shot_list #Jy2/sr'] = p_snlist
            #------
            hist, edges = np.histogram( freq_obs, bins = freqs_edges.to(u.GHz).value, weights = cat[f"I{line}"]/vdelt)
            Ilist = hist * u.Jy / field_size.to(u.sr)
            dict['I_list #Jy/sr']= Ilist
            #------
            cat_galaxies = cat.loc[cat["Mstar"] >= 1e10]
            freq_obs_g = (rest_freq.value / (1 + cat_galaxies['redshift']))
            vdelt = (cst.c * 1e-3) * dnu / freq_obs_g #km/s
            #------        
            hist, edges = np.histogram( freq_obs_g, bins = freqs_edges.to(u.GHz).value)
            snlist = 1 / (hist / field_size.to(u.sr)).value
            dict["gal_shot_list #sr"]=snlist
            #------        
            hist_I, edges = np.histogram( freq_obs_g, bins = freqs_edges.to(u.GHz).value, weights =cat_galaxies[f"I{line}"]/vdelt)
            I_X =  hist_I * u.Jy / field_size.to(u.sr)
            sn_lineshotlist = I_X * snlist
            dict["LIMgal_shot #Jy/sr"]= sn_lineshotlist

            fig, (axI, axSN, axG, axJG) = plt.subplots(1,4,figsize=(16,4), dpi=200)
            for i, (ax, value, unit) in enumerate(zip((axSN, axI, axG, axJG),
                                                (dict['LIM_shot_list #Jy2/sr'],dict['I_list #Jy/sr'],dict["gal_shot_list #sr"],dict["LIMgal_shot #Jy/sr"]),
                                                ('Shot noise [$\\rm Jy^2/sr$]', 'Background intensity [$\\rm Jy/sr$]', 'Galaxy shot noise [$\\rm sr$]', 'Cross-shot noise [$\\rm Jy/sr$]'))):
                x = freq_list / 1e9
                ax.plot(x, value)
                ax.set_xlabel('frequency [GHz]')
                ax.set_ylabel(unit)
                ax.set_yscale('log')
                if(i==2): ax.set_title(f"{catname}_{line}")

            fig.tight_layout()
            fig.savefig('figures/{catname}_{line}_I_and_shotnoises.png', transparent=True)
            plt.close()
            
            #------
            '''
            bias_line_t10 = []
            bias_t10 = []
            angular_k_list = []
            p2d_list = []
            Dc_list = []
            delta_Dc_list = []
            #for each channel:

            for f, (freq,z) in enumerate(zip(freq_list.to(u.GHz).value, z_list)):
                subcat = cat.loc[np.abs(freq_obs-freq) <= dnu/2]

                if(z<0 or len(subcat)==0 or (not 'CII' in line)): 
                    bias_line_t10.append(0)
                    bias_t10.append(0)
                    angular_k_list.append(0)
                    p2d_list.append(0)
                    Dc_list.append(0)
                    delta_Dc_list.append(0)
                else: 
                    #select sources in the channel
                    freq_obs_subcat = (rest_freq.value / (1 + subcat['redshift']))
                    vdelt = (cst.c * 1e-3) * dnu / freq_obs_subcat #km/s
                    #-------------------------------------------------------------------------------------------
                    #b eff using T10
                    bias_subcat = bias.haloBias(subcat["Mhalo"]/h, model = 'tinker10', z=z, mdef = '200m')
                    bias_line_t10.append( np.sum(bias_subcat * subcat[f'I{line}']) / np.sum(subcat[f'I{line}']) )
                    #-------------------------------------------------------------------------------------------
                    subcat = subcat.loc[subcat["Mstar"] >= 1e10]
                    #-------------------------------------------------------------------------------------------
                    #b using T10
                    bias_subcat = bias.haloBias(subcat["Mhalo"]/h, model = 'tinker10', z=z, mdef = '200m')
                    bias_t10.append( np.mean(bias_subcat) )

                    angular_k, p2d, Dc, delta_Dc =  get_2d_pk_matter(z, freq*u.GHz, dnu)

                    angular_k_list.append(angular_k)
                    p2d_list.append(p2d)
                    Dc_list.append(Dc)
                    delta_Dc_list.append(delta_Dc)
                    
                    #-------------------------------------------------------------------------------------------
            dict["beff_t10"] = np.asarray(bias_line_t10)
            dict["beff_gal_t10"] = np.asarray(bias_t10)
            dict['k_angular']= angular_k_list
            dict['pk_matter_2d']= p2d_list
            dict["Dc"]= np.asarray(Dc_list)
            dict["delta_Dc"]= np.asarray(delta_Dc_list)
            '''
        if(not dico_exists): 
            print("save the dict "+f'dict_dir/{catname}_{line}.p')
            pickle.dump(dict, open(dict_pks_name, 'wb'))
        else: 
            print("update the dict"+f'dict_dir/{catname}_{line}.p')
            dico_loaded.update(dict)
            pickle.dump(dico_loaded, open(dict_pks_name, 'wb'))

    else: print('load the dict'+f'dict_dir/{catname}_{line}.p')

    dict = pickle.load( open(dict_pks_name, 'rb'))

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
            #if(tile_sizeRA != 0.2 and tile_sizeRA != 1.25 and l !=76): continue

            for z_center, dz in zip(TIM_params['z_centers'], TIM_params['dz']): 

                dictl[f'pk_3D_z{z_center}_CII_de_Looze'] = p_of_k_for_comoving_cube(file[:-5],'CII_de_Looze',z_center, TIM_params)

            dict_fieldsize[f'{l}'] = dictl

        pickle.dump(dict_fieldsize, open(TIM_params['output_path']+f'pySIDES_from_uchuu_{tile_sizeRA}_x_{tile_sizeDEC}.p', 'wb'))

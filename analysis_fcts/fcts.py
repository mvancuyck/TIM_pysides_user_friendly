import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from IPython import embed
from astropy.cosmology import Planck15 as cosmo
import camb
import pickle
from astropy import constants
import scipy.constants as cst
from scipy.io import readsav
from scipy.interpolate import interp2d, interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from astropy import wcs
import powspec
from astropy.table import Table
import os
import vaex as vx
import time
from scipy.optimize import curve_fit
from functools import partial
from multiprocessing import Pool, cpu_count
from colossus.lss import bias
from colossus.cosmology import cosmology
Cosmo = cosmology.setCosmology('planck15')
h = Cosmo.H0/100
#from astropy.cosmology import z_at_value
import datetime
from astropy.stats import gaussian_fwhm_to_sigma
import astropy.convolution as conv
from matplotlib.colors import LogNorm
import time
from analysis_fcts.set_k import * 
from analysis_fcts.scientific_notation import * 
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler
from scipy.interpolate import interp1d
#-------------------------------------------------------

def powspec_LIMgal(params, line, galaxy, path, J,  z, dz, n, field_size, dkk,
                   recompute=False, to_emb=False ):

    #----
    dict_pks_name = f'dict_dir/dict_LIMgal_{params}.p'
    dico_exists = os.path.isfile(dict_pks_name)
    key_exists = False
    if(dico_exists): 
        dico_loaded = pickle.load( open(dict_pks_name, 'rb'))
        key_exists = ('k' in dico_loaded.keys())
    #--- 
    if(not key_exists or recompute):

        print("Computes the power spectra")
        if(to_emb): embed()

        LIM = fits.getdata(f"{path}/{line}_MJy_sr.fits")*u.MJy/u.sr
        hdr = fits.getheader(f"{path}/{line}_MJy_sr.fits")
        w = wcs.WCS(hdr)
        res = hdr["CDELT1"]* u.Unit(hdr["CUNIT1"]) #w.wcs.cdelt[0] * u.deg
        npix = hdr["NAXIS1"] * hdr["NAXIS2"]

        k_nyquist, k_min, delta_k, k_bintab, k, k_map = set_k_infos(hdr["NAXIS2"],hdr["NAXIS1"], res, delta_k_over_k = dkk)
        
        pk_im_1d = []
        for i in range(LIM.shape[0]):
            pk, _ = my_p2(LIM[i,:,:], res, k_bintab, u.Jy**2/u.sr, u.arcmin**-1 )
            pk_im_1d.append( pk.value ) 
        pk_im_1d = np.asarray(pk_im_1d)

        #---
        Gal =  fits.getdata(f"{path}/{galaxy}_pix.fits")/u.sr

        pk_imB_1d = []
        pk_p2 = []
        for i in range(LIM.shape[0]):
            #----x
            gal_mean =  Gal[i,:,:].mean()
            if(gal_mean <0): continue
            else: Gal[i,:,:] /= gal_mean.value
            #----            
            p_imB, _ = my_p2(Gal[i,:,:], res, k_bintab, u.sr**-1, u.arcmin**-1 )
            pk_imB_1d.append(p_imB.value)
            #----
            pk, _    = my_p2(Gal[i,:,:], res, k_bintab, u.Jy*u.sr**-1, u.arcmin**-1, map2 = LIM[i,:,:])

            pk_p2.append(pk.value)
        pk_imB_1d = np.asarray(pk_imB_1d)       
        pk_p2 = np.asarray(pk_p2)
        
        dico = {'k': k.to(u.arcmin**-1),
                'kbin':k_bintab,
                'deltak':delta_k,
                'res':res.to(u.rad),
                'pk_J':         (np.mean(pk_im_1d,  axis = 0)*u.Jy**2/u.sr, np.std(pk_im_1d,  axis = 0)*u.Jy**2/u.sr),
                'pk_gal':       (np.mean(pk_imB_1d, axis = 0)/u.sr,         np.std(pk_imB_1d, axis = 0)/u.sr),
                'pk_J-gal':     (np.mean(pk_p2, axis = 0)*u.Jy/u.sr,        np.std(pk_p2, axis = 0)*u.Jy/u.sr),
                'nb_pk_gal_computed': len(pk_imB_1d)}

        for l in (range(len(pk_imB_1d))):
            dico[f'pk_J_{l}'] = pk_im_1d[l]*u.Jy**2/u.sr
            dico[f'pk_gal_{l}'] = pk_imB_1d[l]/u.sr
            dico[f'pk_J-gal_{l}'] = pk_p2[l]*u.Jy/u.sr

        if(not dico_exists): 
            print('save the dict')
            pickle.dump(dico, open(dict_pks_name, 'wb'))
        else: 
            print('update the dict')
            dico_loaded.update(dico)
            pickle.dump(dico_loaded, open(dict_pks_name, 'wb'))

    print("load the dict")
    dict = pickle.load( open(dict_pks_name, 'rb'))

    return dict

#-------------------------------------------------------

def compute_other_linear_model_params( params, line, path, J, rest_freq, z, dz, n_slices, field_size, cat, recompute=False, to_emb=False):

    #----
    dict_pks_name = f'dict_dir/dict_LIMgal_{params}.p'
    dico_exists = os.path.isfile(dict_pks_name)
    key_exists = False
    if(dico_exists): 
        dico_loaded = pickle.load( open(dict_pks_name, 'rb'))
        key_exists = ('LIM_shot' in dico_loaded.keys())
    
    if(not key_exists or recompute):
        
        if(to_emb): embed()
        
        print(f"Computes the dict")
        if(cat is None): 
            print("load the cat plz")
            return 0
        # ------
        nu_obs = rest_freq / (1+z)
        hdr = fits.getheader(f"{path}/{line}_MJy_sr.fits")
        res = hdr['CDELT1'] * u.Unit(hdr['CUNIT1'])
        field_size = ( hdr["NAXIS1"] * hdr["NAXIS2"] * res**2 ).to(u.deg**2)
        w = wcs.WCS(hdr)    
        #index of channel containing freq_obs
        freq_i = np.round( w.swapaxes(0, 2).sub(1).wcs_world2pix(nu_obs.to(u.Hz),  0)[0] ).astype(int)    
        dnu = np.round( (hdr["CDELT3"] * u.Unit(hdr["CUNIT3"])).to(u.GHz).value, 2)
        nu = np.round(hdr["CRVAL3"] * u.Unit(hdr["CUNIT3"]).to(u.GHz),5) + np.round(hdr["CDELT3"] * u.Unit(hdr["CUNIT3"]).to(u.GHz),5)*freq_i
        #central frequencies of channels around the central channel, +- n_average
        freqs = np.linspace(nu-n_slices*dnu, nu+n_slices*dnu, int(2*n_slices+1))
        #edges of channels 
        freqs_edges = np.linspace(nu-n_slices*dnu-dnu/2, nu+n_slices*dnu+dnu/2, int(2*n_slices+2))
        nu_max_edge = freqs_edges.max(); nu_min_edge = freqs_edges.min()
        #corresponding redshifts of central frequencies.
        z_centers = rest_freq.value / freqs - 1
        #------
        angular_k, p2d, Dc, delta_Dc =  get_2d_pk_matter(z, nu_obs, dnu)
        #------
        cat = cat.loc[ (np.abs(rest_freq.value/(1+cat['redshift'])-nu_obs)) <= (nu_max_edge - nu_min_edge)/2]
        freq_obs = (rest_freq.value / (1 + cat['redshift']))
        vdelt = (cst.c * 1e-3) * dnu / freq_obs #km/s
        #------
        hist, edges = np.histogram( rest_freq.value / (1+cat["redshift"]) , bins = freqs_edges, weights = (cat[f"I{J}"]/vdelt)**2)
        p_snlist = hist * u.Jy**2 / field_size.to(u.sr) 
        p_sn = (p_snlist.mean(), p_snlist.std())
        #------
        hist, edges = np.histogram( rest_freq.value / (1+cat["redshift"]), bins = freqs_edges, weights = cat[f"I{J}"]/vdelt)
        Ilist = hist * u.Jy / field_size.to(u.sr)
        I = (Ilist.mean(), Ilist.std())
        #------
        cat_galaxies = cat.loc[cat["Mstar"] >= 1e10]
        freq_obs = (rest_freq.value / (1 + cat_galaxies['redshift']))
        vdelt = (cst.c * 1e-3) * dnu / freq_obs #km/s
        #------        
        hist, edges = np.histogram( rest_freq.value / (1+cat_galaxies["redshift"]), bins = freqs_edges)
        snlist = 1 / (hist / field_size.to(u.sr)).value
        shot_gal = ( np.asarray(snlist).mean() , np.asarray(snlist).std())
        #------        
        hist_I, edges = np.histogram( rest_freq.value / (1+cat_galaxies["redshift"]), bins = freqs_edges, weights =cat_galaxies[f"I{J}"]/vdelt)
        I_X =  hist_I * u.Jy / field_size.to(u.sr)
        sn_lineshotlist = I_X * snlist
        shot_cross_gal = (sn_lineshotlist.mean(), sn_lineshotlist.std())
        #------
        bias_line_t10 = []
        bias_t10 = []
        #for each channel:
        '''
        for i in range(len(freqs)):
            #select sources in the channel
            subcat = cat.loc[np.abs(rest_freq.value/(1+cat['redshift'])-freqs[i]) <= dnu/2]
            freq_obs = (rest_freq.value / (1 + subcat['redshift']))
            vdelt = (cst.c * 1e-3) * dnu / freq_obs #km/s
            #-------------------------------------------------------------------------------------------
            #b eff using T10
            bias_subcat = bias.haloBias(subcat["Mhalo"]/h, model = 'tinker10', z=z, mdef = '200m')
            bias_line_t10.append( np.sum(bias_subcat * subcat[f'I{J}']) / np.sum(subcat[f'I{J}']) )
            #-------------------------------------------------------------------------------------------
            subcat = subcat.loc[subcat["Mstar"] >= 1e10]
            #-------------------------------------------------------------------------------------------
            #b using T10
            bias_subcat = bias.haloBias(subcat["Mhalo"]/h, model = 'tinker10', z=z, mdef = '200m')
            bias_t10.append( np.mean(bias_subcat) )
            #-------------------------------------------------------------------------------------------
        beff_t10=( np.asarray(bias_line_t10).mean(), np.asarray(bias_line_t10).std() )
        beff_gal_t10 = ( np.asarray(bias_t10).mean(), np.asarray(bias_t10).std() )
        #------
        '''
        dict = {'species':J,
                'z':z, 'dz':dz,
                'cube':f"{path}/{line}_MJy_sr.fits",   
                'I':I, 'Ilist':Ilist,
                'LIM_shot':p_sn, 'LIM_shot_list':p_snlist,
                "gal_shot":shot_gal, "gal_shot_list":snlist,
                "LIMgal_shot":shot_cross_gal, "LIMgal_shot_list":sn_lineshotlist,
                'k_angular':angular_k, 'pk_matter_2d':p2d, "Dc":Dc, "delta_Dc":delta_Dc}
        #'beff_t10':beff_t10, 'beff_t10_list':bias_line_t10,
        #"bgal_t10":beff_gal_t10, "bgal_t10_list":bias_t10,
        
        if(not dico_exists): 
            print("save the dict")
            pickle.dump(dict, open(dict_pks_name, 'wb'))
        else: 
            print("update the dict")
            dico_loaded.update(dict)
            pickle.dump(dico_loaded, open(dict_pks_name, 'wb'))
    print('load the dict')
    dict = pickle.load( open(dict_pks_name, 'rb'))

    return dict 
# -------------------------------------------------------------------------------------------------------------

def gen_linear_noise(freqs, dnu, pix_size, pixel_sr, ny,nx, sensitivity_file = '/home/mvancuyck/sides/PYSIDES/pysides/CONCERTO_SENSITIVITY/SIDES_sensitivity_1Ghz_res.txt', reduction_factor = 1):
    #load the sensitivity file
    nu_file, sens_mJy_per_beam, omega_beam_sensfile = np.loadtxt(sensitivity_file, unpack = True)
    sens_mJy_per_beam *= np.sqrt(1/dnu)
    sigma_Jy_per_beam = np.interp(freqs, nu_file, 1.e-3 * sens_mJy_per_ [0.5*delta_ra.value / pix_resol, 0.5*delta_dec.value / pix_resol, 1])
    noise_cube = []
    for freq, sigma_1pix_Jy_beam in zip(freqs, sigma_Jy_per_beam):
        fwhm = 1.22 * cst.c / (freq * 1e9 * telescope_diameter) * u.rad
        sigma = (fwhm * gaussian_fwhm_to_sigma).to(u.arcsec)
        sigma_pix = sigma.value / (pix_size)  #pixel
        kernel = conv.Gaussian2DKernel(x_stddev=sigma_pix, x_size=nx)
        kernel.normalize(mode="peak")
        norm_psf =  np.sqrt(np.sum(kernel.array**2))
        noise_slice = np.random.normal(size = (ny,nx)) * sigma_1pix_Jy_beam * 1.e-6 / pixel_sr / reduction_factor / norm_psf
        noise_cube.append(noise_slice)
    return np.asarray(noise_cube)


def get_2d_pk_matter(z, nu_obs, dnu):
    
    pars = camb.read_ini("planck2018.ini")
    pars.set_matter_power(redshifts=[z], kmax=100)
    results = camb.get_results(pars)    
    #Linear spectra
    #_,_,pk_matter = results.get_linear_matter_power_spectrum(hubble_units=False, k_hunit= False)
    k,_,pk_matter_nonlin = results.get_nonlinear_matter_power_spectrum(hubble_units=False, k_hunit= False)

    Dc = cosmo.comoving_distance(z) / u.rad
    delta_Dc = ( (cst.c*1e-3*u.km/u.s) * (1+z) * dnu*u.GHz / cosmo.H(z) / nu_obs.to(u.GHz)) # == cosmo.comoving_distance(z+dzmax) - cosmo.comoving_distance(z-dzmin) checked
    angular_k = (Dc * k/u.Mpc).to(u.arcmin**-1)/2/np.pi #(Dc.value * k_per_mpc.value) * (np.pi/180/60) * (u.arcmin**-1) /2/np.pi checked
    p2d = pk_matter_nonlin[0] / (Dc**2*delta_Dc).value

    return angular_k, p2d, Dc, delta_Dc

def cosmo_distance(z, dnu, nu_obs):
    dz = dnu * (1+z) / nu_obs
    Dc_min = cosmo.comoving_distance(z-dz/2)
    Dc_max = cosmo.comoving_distance(z+dz/2)
    Dc =  cosmo.comoving_distance(z)/u.rad
    delta_Dc = ( (cst.c*1e-3*u.km/u.s) * (1+z) * dnu / cosmo.H(z) / nu_obs)
    pk_3d_to_2d = 1/(Dc**2*delta_Dc)
    k_3d_to_2d  = Dc/2/np.pi
    full_volume_at_z = 4/3*np.pi*(Dc_max**3-Dc_min**3)
    return Dc, delta_Dc, pk_3d_to_2d.to(u.sr/u.Mpc**3), k_3d_to_2d, full_volume_at_z


def volume(omega, Dc, z, nu_obs, dnu, full_volume_at_z=None):
    c = cst.c*1e-3*u.km/u.s
    y = (c / cosmo.H(z)) * (1+z)**2  / (nu_obs.to(u.GHz)*(1+z))
    if(full_volume_at_z is None): volume = (Dc**2).to(u.Mpc**2/u.sr)*(omega).to('sr') * y * dnu
    else:                         volume = omega.to(u.sr) * full_volume_at_z/4/np.pi/u.sr
    return volume

def beam(FWHM):
    sigma_beam = FWHM * gaussian_fwhm_to_sigma
    Omega_beam = (2*np.pi *sigma_beam**2).to(u.sr)
    return Omega_beam

def sigma(P_A, N_modes, P_B=None, P_AB=None):
    if(P_B is None):  P_B  = P_A
    if(P_AB is None): P_AB = np.sqrt(P_A*P_B)
    sigma = np.sqrt(P_AB**2 + (P_A*P_B) ) / np.sqrt(2*N_modes)
    return sigma

def SNR_fct(signal_3d, sigma_3d):
    SNR = signal_3d/sigma_3d
    SNR_comb = np.abs(np.sqrt(np.sum(SNR**2)))
    return  SNR_comb

def gaussian_pk(k, sigma):
    return np.exp(- (np.pi * k)**2 * (2 * sigma**2))**2

def inverse(x):
    freq_CII = 1900.53690000
    return -1 + (freq_CII/x)

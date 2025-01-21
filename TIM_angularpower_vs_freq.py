import matplotlib.pyplot as plt
#plt.ion()
import numpy as np
from powspec import power_spectral_density
import astropy.constants as cst
from astropy.stats import gaussian_fwhm_to_sigma
import astropy.convolution as conv
from IPython import embed  # for debugging purpose only
from astropy.io import fits
from astropy import units as u
from functools import partial
import pickle
from set_k import *

def inverse(x):
    freq_CII = 1900.53690000 
    return -1 + (freq_CII/x)

D = 12 * u.m

if __name__ == "__main__":

    #Load cubes
    path_original = "OUTPUT_TIM_CUBES_FROM_UCHUU/"
    prefix_original = "pySIDES_from_uchuu_TIM_tile# #0_0.2deg_1deg_res20arcsec_dnu4.0GHz_CII_de_Looze_smoothed_MJy_sr.fits"
    units = "MJy_sr"
    beam = "nobeam"
    resfile = "20arcsec"
    res = 20*u.arcsec
    dnu = "4.0GHz"
    freq_CII = 1900.53690000
    n = 120
    #Set frequency axis info
    freqs = np.arange(715,1250,4) 

    #k_nyquist, k_min, k_bin_width, k_bin_tab, k_out, k_map, histo = set_k_infos(nx,ny, res, 0.1)
    #pkg_list = np.zeros((n,6,len(freqs),len(k_out)))

    dictfile = 'TIM_pkgs.p'
    if(not os.path.isfile(dictfile)):
        for tile in range(n):
            for i, species in enumerate('CO_all', 'CI_both', 'CII_de_Looze', 'all_lines_de_Looze', 'continuum', 'full_de_Looze'):
                cube = fits.getdata(f"{path_original}/{prefix_original}{tile}_0.2deg_1deg_res{resfile}_dnu{dnu}_{species}_{beam}_{units}.fits")
                embed()
                mean = cube.mean(axis=(1,2))
                cube -= mean[:, np.newaxis, np.newaxis]
                for f, F in enumerate(freqs):
                    pk, e =  power_spectral_density(cube[f,:,:]*u.MJy/u.sr, res, bins = k_bin_tab)
                    pkg_list[tile, i, f, :] = pk
        
        pickle.dump(pkg_list, open(dictfile, 'wb'))
    else: dict = pickle.load( open(dictfile, 'rb'))

    k = -2
    k_label = np.round( k_bin_tab[k], 2)
    #Figure Pk(@fix scale) vs freq
    BS = 8; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS); mk = 3; l=1
    fig, ax = plt.subplots(1,1, figsize=(4,3), dpi=200)
    
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel(f"P(k)@({k_label}"+" $\\rm arcmin^{-1}$) [$\\rm Jy^2$/sr]")
    ax.set_yscale('log')
    ax.set_xscale('linear')
    ax.set_ylim(1e-1, 1e2)
    ax.set_xlim(237,305)
    def g(freq_CII, x): return -1 + (freq_CII/x)
    Z = partial(g, freq_CII)    
    secax = ax.secondary_xaxis('top', functions=(Z,Z))
    secax.set_xlabel('redshift [CII]')
    plt.legend(loc="center right", fontsize=BS)# bbox_to_anchor=(1.05, 0.0), fontsize=BS);
    plt.tight_layout()
    [plt.savefig(f"pk_at_fix_scale_vs_freq_tim.{extension}") for extension in ("png", "pdf") ]


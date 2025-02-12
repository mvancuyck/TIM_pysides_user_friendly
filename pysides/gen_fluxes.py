from time import time
import numpy as np
import pandas as pd
from astropy import units as u
import scipy.constants as cst
import pickle
import os
from astropy.cosmology import Planck15 as cosmo
from IPython import embed
from pathlib import Path
from astropy.utils.console import ProgressBar
from astropy.stats import gaussian_fwhm_to_sigma

from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import zip_longest
from progress.bar import Bar
from IPython import embed
import time

def gen_fluxes(cat, params):

    tstart = time()

#quantity required in the cat pandas input: redshift, issb (True if starburst), SFR

    print("Generate SED properties and fluxes...")

    #compute up to which z the SBs evolve

    zlimSB = (np.log10(params['UmeanSB']) - np.log10(params['UmeanMSz0'])) / params['alphaMS']  # When the flat low-z USB value > UMS(z), USB = UMS, zlimSB is the redhift at which it happens

    if zlimSB > params['zlimMS']:
        print("zlim SB (when UMS = USB)= {}".format(params['zlimSB']))
        zlimSB = 9999.0  # USB will be constant at all z (the SB curve never cross the MS one!)

        
    if not ("Dlum" in cat.columns):
        print('Compute luminosity distances since they have not been computed before...')
        cat = cat.assign(Dlum = cosmo.luminosity_distance(cat["redshift"]).value) #Mpc

    Ngal = len(cat)

    print('Draw <U> parameters...')

    Umean = np.zeros(Ngal)

    #attribute a <U> value to each galaxy

    index_MS_SBhighz = np.where( (cat["issb"] == False) | (cat["redshift"] >= zlimSB) )

    Umean[index_MS_SBhighz[0]] = 10.**( np.log10(params['UmeanMSz0']) + params['alphaMS'] * np.minimum(cat["redshift"][index_MS_SBhighz[0]], params['zlimMS']) )

    index_SBlowz = np.where( (cat["issb"] == True) & (cat["redshift"] < zlimSB) )

    Umean[index_SBlowz[0]] = params['UmeanSB']

    #add log-normal scatter

    Umean *= 10.**np.random.normal(scale = params['sigma_logUmean'], size = Ngal)

    cat = cat.assign(Umean = Umean)

    print('Load SED and LIR grids...')

    SED_dict = pickle.load(open(params['SED_file'], "rb"))

    LIR_LFIR_ratio_dict = pickle.load(open(params['ratios_file'], "rb"))

    print("Generate LIR...")
    cat = cat.assign(LIR = params['SFR2LIR']/1.5 * cat["SFR"]) #assume full IR reprocessing!

    print("Generate monochromatic fluxes...")
    Snu_arr = gen_Snu_arr(params['lambda_list'], SED_dict, cat["redshift"], cat['mu']*cat["LIR"], cat["Umean"], cat["Dlum"], cat["issb"]) 

    for i in range(0,len(params['lambda_list'])): #option for cubes: pas d'assigne else gensnuarr sum=true
        kwargs = {'S{:d}'.format(params['lambda_list'][i]) : Snu_arr[:,i].value}
        cat = cat.assign(**kwargs)

    #generate LFIR (40-400 microns)
    print("Generate LFIR...")
    cat = cat.assign(LFIR = gen_LFIR_vec(LIR_LFIR_ratio_dict, cat["redshift"], cat["LIR"], cat["Umean"], cat["issb"]))

    tstop = time()

    print('SED properties of ', len(cat), ' generated in ', tstop-tstart, 's')

    return cat

### Add fluxes allows to add some wavelnegth a posteriori


def gen_Snu_arr(lambda_list, SED_dict, redshift, LIR, Umean, Dlum, issb):

    #mask = np.where((SED_dict["lambda"]>= lambda_rest[j,0]) & (SED_dict["lambda"] <= lambda_rest[j,1]))

    stype = ["nuLnu_SB_arr" if a else "nuLnu_MS_arr" for a in issb]
    Uindex = np.round((Umean - SED_dict["Umean"][0]) / SED_dict["dU"])
    Uindex = Uindex.astype(int)
    Uindex = np.maximum(Uindex, 0)
    Uindex = np.minimum(Uindex, np.size(SED_dict["Umean"]) - 1)

    lambda_rest = lambda_list / (1 + np.array(redshift)[:, np.newaxis]) * u.um #lambda list is in micron!
    nu_rest_Hz = (cst.c * u.m/u.s) / lambda_rest.to(u.m)
    nuLnu     = np.zeros([len(redshift), len(lambda_list)])

    for j,k in enumerate(Uindex):
        nuLnu[j,:] = np.interp(lambda_rest[j,:].value, SED_dict["lambda"], SED_dict[stype[j]][k]) 

    nuLnu /= nu_rest_Hz.value
    Lnu = (3.828e26 * u.W) * np.array(LIR)[:, np.newaxis] * nuLnu / u.Hz #W/Hz (the output of the worker is in Hz^-1)
    Numerator = Lnu * ( 1 + np.array(redshift)[:,np.newaxis]) * (1/ (np.pi *  4 ))
    Denominator = ((np.asarray(Dlum) * u.Mpc).to(u.m)) ** 2
    Snu_arr = ( Numerator / Denominator[:, np.newaxis] ).to(u.Jy)

    return Snu_arr  

def add_fluxes(cat, params, new_lambda):

    tstart = time()

    SED_dict = pickle.load(open(params['SED_file'], "rb"))

    print("Add new monochromatic fluxes...")
    Snu_arr = gen_Snu_arr(new_lambda, SED_dict, cat["redshift"], cat['mu']*cat["LIR"], cat["Umean"], cat["Dlum"], cat["issb"]) 

    for i in range(0,len(new_lambda)): #option for cubes: pas d'assigne else gensnuarr sum=true
        kwargs = {'S{:d}'.format(new_lambda[i]) : Snu_arr[:,i].value}
        
        cat = cat.assign(**kwargs)

    tstop = time()

    print('New fluxes of ', len(cat), ' galaxies generated in ', tstop-tstart, 's')

    return cat

###routine to compute Snu_arr, can be used outside of the catalog generation to produce intensity mapping cubes on the fly without filling the memery with hundreds of wavelengths in un the catalog

def gen_LFIR_vec(LIR_LFIR_ratio_dict, redshift, LIR, Umean, issb):

    LFIR = np.zeros_like(redshift)

    selSB = np.where(issb == True)
    selMS = np.where(issb == False)

    Uindex = np.round((Umean - LIR_LFIR_ratio_dict["Umean"][0]) / LIR_LFIR_ratio_dict["dU"])
    Uindex = Uindex.astype(int)
    Uindex = np.maximum(Uindex, 0)
    Uindex = np.minimum(Uindex, np.size(LIR_LFIR_ratio_dict["Umean"]) - 1)

    LFIR[selSB[0]] = LIR[selSB[0]] * LIR_LFIR_ratio_dict["LFIR_LIR_ratio_SB"][Uindex[selSB[0]]]
    LFIR[selMS[0]] = LIR[selMS[0]] * LIR_LFIR_ratio_dict["LFIR_LIR_ratio_MS"][Uindex[selMS[0]]]

    return LFIR



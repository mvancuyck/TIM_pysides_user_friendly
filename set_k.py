#Imports
import numpy as np
import scipy.constants as cst
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from IPython import embed
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from scipy import interpolate
import camb
from camb import get_matter_power_interpolator
from camb import model, initialpower
import pandas as pd

def my_p2(data_map, res, k_bins, unitpk, unitk, map2 = None, linearbinTrue_logbinFalse = False):
    """
    Measure the power spectrum  in a map

    data_map (astropy.Quantity): the map in which to measure the power spectum
    res (astropy.Quantity): the resolution of the map
    map2 (optional, astropy.Quantity): the second map in case of cross correlation

    return:
    
    p2 (astropy.Quantity): the measured power spectrum
    """
    
    ny,nx = data_map.shape
    npix = ny*nx
    norm =  ((res.to(u.rad))**2).to(u.sr) / npix

    k_map  =  give_map_spatial_freq(res, ny, nx) #rad-1
    
    if(map2 is not None):
        ft  = np.fft.fft2( data_map)
        ft1 = np.fft.fft2( map2)
        p2  = np.real(0.5*( (ft*np.conj(ft1)) + (np.conj(ft)*ft1) ))  * norm
    else: p2 = np.abs( (np.fft.fft2( data_map))**2) * norm
    
    histo, edegs = np.histogram(k_map, bins = k_bins)
    pk_1d, edges = np.histogram(k_map, bins = k_bins, weights = p2)
    pk_1d /= histo

    if linearbinTrue_logbinFalse: k = k_bins[:-1] + np.diff(k_bins)/2
    else: 
        k, edges = np.histogram(k_map, bins = k_bins, weights = k_map)
        k /= histo
    
    return pk_1d.to(unitpk), k.to(unitk)

def give_map_spatial_freq(res, ny, nx):
    
    res = res.to(u.rad)
    Ys, Xs = np.unravel_index(np.arange(nx*ny),(ny,nx))
    N = np.zeros((ny, nx))
    M = np.zeros((ny, nx))
    for  xs, ys, in zip(Xs, Ys):
        N[ys,xs] = int(xs)
        M[ys,xs] = int(ys)
    w_sup = np.where(M>ny/2)
    M[w_sup] = ny - M[w_sup]
    w_sup = np.where(N>nx/2)
    N[w_sup] = nx - N[w_sup]
    Kx = N/nx/res
    Ky = M/ny/res
    k_map = (Kx**2 + Ky**2)**(1/2)
        
    return k_map

def make_bintab(k, delta_k_min, dkk = 0, delta_k_max=0, ):
    kmax      = k[1].value
    kmin      = k[0].value
    if(dkk == 0): 
        bintab = np.arange(kmin, kmax, delta_k_min.value )
        bin_width = np.ones(len(bintab)-1) * delta_k_min.value
    else:
        k1 = kmin
        delta_k = 0 
        bintab = []
        bintab.append(kmin)
        bin_width = []
        while(k1 + delta_k <= kmax):
            delta_k = np.minimum( np.maximum(k1*dkk, delta_k_min.value) , (kmax - k1) )
            if(delta_k_max != 0): 
                delta_k = np.minimum(delta_k, delta_k_max.value)
            k1 = k1 +  delta_k
            bintab.append(k1)
            bin_width.append(delta_k)
        bintab = np.asarray(bintab) 
    if( bintab.max() <= kmax):
        bintab[-1]    = kmax
        bin_width[-1] = bintab[-1]-bintab[-2]
    #bintab = np.insert(bintab,0,0)
    return bintab * k[0].unit , bin_width * k[0].unit
    
def set_k_infos(ny, nx, res, delta_k_over_k = 0):

    #compute the maximum value of l
    k_nyquist = 1 / 2 / res.to(u.rad)  #rad**-1
    #compute the map of the radial l multipols
    k_map  =  give_map_spatial_freq(res, ny, nx) #rad-1
    #Measuring from the map of l the miminum value of l
    k_min = np.min(k_map[np.nonzero(k_map)]) #rad-1
    #Computing the minimum size of a l bin
    #dk_min =  1 / (np.min((nx,ny)) * res.to(u.rad)) #rad-1
    dk_min = 2 / ( np.minimum(ny,nx) * res.to(u.rad)) #rad-1

    k_range = [k_min, k_nyquist] #rad**-1
    #Setting the binning of the l map
    k_bin_tab, k_bin_width = make_bintab(k_range, dk_min, delta_k_over_k) #rad-1 
    #Compute bin addresses
    k_out, edges = np.histogram(k_map, bins = k_bin_tab, weights = k_map)
    histo, edegs = np.histogram(k_map, bins = k_bin_tab)
    k_out = k_out / histo
    return k_nyquist, k_min, k_bin_width, k_bin_tab, k_out, k_map


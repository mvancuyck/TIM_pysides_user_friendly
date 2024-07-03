'''
This program computes the expected sensitivity and mapping speed for CONCERTO (before on-sky measurements).
The sensivity is derived from the NIKA2 on-sky sensitivity. NIKA2 is a KIDS camera, mounted on the IRAM 30m telescope. It has been built by the CONCERTO team.
All computations and assumptions are detailed in the CONCERTO collaboration paper: 
"A wide field-of-view low-resolution spectrometer at APEX: Instrument design and scientific forecast", 2020, A&A 642, 60.
Enter your parameters in the "concerto_param" dictionnary.   
This program gives also specific numbers for the [CII] line. For other lines, you need to modify the "zcii" variable.

Written by Guilaine Lagache and used in Concerto collaboration et al. (2020, A&A, 642, A60). Partially modified by Matthieu Bethermin for pySIDES.
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from astropy.cosmology import Planck18_arXiv_v2 as cosmo
import astropy.units as u
import astropy.constants as cst
from astropy.stats import gaussian_fwhm_to_sigma
from IPython import embed

# Do not change nika2_param    
nika2_param = {
    "bandpass": [39.2, 48.] * u.GHz,
    "aperture_diam": 27.5 * u.m,  # size of illumination of the 30m-telescope
    "nu": [150.0, 260.0] * u.GHz, # NIKA2 frequencies
    #"nefd": [9.8, 36.0] * u.mJy * u.s ** (0.5),  # Point source from Perotto+2020, 60deg, pwv=2, table 17
    #"nefd": [7.5, 10.] * u.mJy * u.s ** (0.5),  # Min
    #"nefd": [15., 20.] * u.mJy * u.s ** (0.5),  # Max
    "nefd": [10, 15.0] * u.mJy * u.s ** (0.5),  # Average ; without the current limitation on the dichroic + efficiency + sky noise
}

concerto_param = {
    "bandpass": 115.0 * u.GHz, # bandpass width (it is the same for the two arrays)
    "aperture_diam": 11 * u.m,  # 11m is the illumination of the APEX telescope
    "nu": [211.22557, 237.62877, 271.57574, 316.83836] * u.GHz, # To compute the sensitivity at given nu (i.e. given redshifts, here for [CII])
    "delta_nu": 1.5 * u.GHz, # spectral resolution. Can be between 1 and 10 GHz.
    "fov": 20 * u.arcmin, # size of the FOV (round diameter)
    "nkids": 2150.0 * 80.0 / 100.0, # considering 80% valid pixels
    "survey_area": 1.4 * u.deg * u.deg,  # sq. deg. (survey area has not to be smaller than the FOV)
    "survey_time": 1200 * 0.7 * u.h,  # hours of observation on-source
}

#Concerto paramaters slightly modified for SIDES
sides_param = {
    "bandpass": 115.0 * u.GHz, # bandpass width (it is the same for the two arrays)
    "aperture_diam": 11 * u.m,  # 11m is the illumination of the APEX telescope
    "nu": np.arange(125,306,1) * u.GHz, # To compute the sensitivity at given nu (i.e. given redshifts, here for [CII])
    "delta_nu": 1.0 * u.GHz, # spectral resolution. Can be between 1 and 10 GHz.
    "fov": 20 * u.arcmin, # size of the FOV (round diameter)
    "nkids": 2150.0 * 80.0 / 100.0, # considering 80% valid pixels
    "survey_area": 1.4 * u.deg * u.deg,  # sq. deg. (survey area has not to be smaller than the FOV)
    "survey_time": 1200 * 0.7 * u.h,  # hours of observation on-source
    }

deep_lowres_100h_params = {
    "bandpass": 115.0 * u.GHz, # bandpass width (it is the same for the two arrays)
    "aperture_diam": 11 * u.m,  # 11m is the illumination of the APEX telescope
    "nu": np.arange(125,306,1) * u.GHz, # To compute the sensitivity at given nu (i.e. given redshifts, here for [CII])
    "delta_nu": 18.08 * u.GHz, # spectral resolution. Can be between 1 and 10 GHz.
    "fov": 20 * u.arcmin, # size of the FOV (round diameter)
    "nkids": 2150.0 * 80.0 / 100.0, # considering 80% valid pixels
    "survey_area": 0.25 * u.deg * u.deg,  # sq. deg. (survey area has not to be smaller than the FOV)
    "survey_time": 100 * 0.7 * u.h,  # hours of observation on-source
    }

def sensitivity_dual_band(param, nika2, do_plot=False):
    # Compute CONCERTO sensitivity for continuum as if it was a photometer with the "bandpass_concerto"

    nefd1 = nika2["nefd"] * np.sqrt(2.0) * np.sqrt(2.0) * (nika2["aperture_diam"] / param["aperture_diam"]) ** 2
    # 2 times sqrt(2) because 2 polarisers and 50% lost polar

    # Need to scale for different bandpass between NIKA2 (=42GHz) and CONCERTO:
    nefd2 = nefd1 * np.sqrt(nika2["bandpass"] / param["bandpass"])
    # Assume that we are at the photon noise, in photometric mode and continuum
    # measurement, we gain like sqrt(bandwidth) (Jy are per Hz)

    transmission = 0.8  # just MPI
    nefd3 = nefd2 / np.sqrt(transmission)  # unit = mJy.s1.2
    # Less photons as tranmission => less noise as sqrt(transmission)

    # Interpolate nefd from nu_nika2 frequencies to nu (CONCERTO)
    nu_temp = np.concatenate(([120.0, 140.0] * u.GHz, nika2["nu"], [270.0, 280.0, 310.0, 360.0] * u.GHz))
    nefd_min = np.amin(nefd3)
    nefd_max = np.amax(nefd3)
    nefd_temp = np.concatenate(([nefd_min, nefd_min], nefd3, [nefd_max, nefd_max, nefd_max, nefd_max]))
    linear_interp = interpolate.interp1d(nu_temp, nefd_temp)
    nefd = linear_interp(param["nu"]) * nefd_temp.unit  # linear_interp loose unit
    # Check
    if do_plot:
        plt.clf()
        plt.ion()
        plt.plot(nu_temp, nefd_temp, "k", marker="o", label="NIKA2")
        plt.plot(param["nu"], nefd, color="r", marker="o", label="Interpolation on CONCERTO frequencies")
        plt.ylabel("nefd [mJy s^0.5]", fontsize=13)
        plt.xlabel("Frequency [GHz]", fontsize=13)
        plt.title("NEFD Point source for CONCERTO")
        plt.legend()
        plt.show()

    return nefd


def sensitivity_spectro(param, nefd):
    # Compute the sensitivity per spectral element
    n_spec_el = param["bandpass"] / param["delta_nu"]
    sens_per_spec_el = nefd * n_spec_el  # Sensititivy per spectral element mJy.s1/2
    return sens_per_spec_el


def compute_PS_sensitivity(param):
    # Compute the point source sensitivity (per chanel)
    rms = (
        1.0
        / np.sqrt(param["mapping_speed"])
        * 1.0
        / np.sqrt(param["survey_time"])
        * np.sqrt(param["survey_area"])
        * param["omega_beam"]
    )# in Jy for one chanel delta_nu
    #print("Omega_beam [sr]= ", param["omega_beam"].value)
    #print("rms [mJy]= ", rms.to(u.mJy).value)

    # For the CII line:
    zcii = cst.c / (157.7 * u.micron) / param["nu"] - 1
    #print("Redshifts CII = ", zcii)
    dv = param["delta_nu"] / param["nu"] * cst.c #m/sec
    # Line luminosity 
    L_line = 4 * np.pi / cst.c * (rms * dv) * cosmo.luminosity_distance(zcii) ** 2 * param["nu"]
    L_line = L_line.to(10 ** 9 * u.L_sun)
    # Alternatively use Carilli & Walter (2013) -- same result L_sol directly
    # L_line = 1.04e-3 * rms * (dv * 1.e-3)  * (cosmo.luminosity_distance(zcii))**2. * param["nu"]/1.e9
    return rms, L_line

def derive_useful_param(param):
    nefd = sensitivity_dual_band(param, nika2_param)
    nefd_spectro = sensitivity_spectro(param, nefd)
    fwhm_beam = 1.22 * u.rad * cst.c / (param["nu"]) / param["aperture_diam"]
    sigma_beam = fwhm_beam * gaussian_fwhm_to_sigma
    beam_area = 2.0 * np.pi * sigma_beam ** 2
    beam_area = beam_area.to(u.sr)
    nefd_spectro_per_beam = nefd_spectro / beam_area  # mJy.s1/2 / sr
    nefd_spectro_per_beam = nefd_spectro_per_beam.to(u.MJy * u.s**0.5 /u.sr)   # MJy.s1/2 / sr
    
    # Compute naive mapping speed (one beam = one pixel)
    sigma_array = nefd_spectro_per_beam / np.sqrt(param["nkids"])  # on sky sensitivity [MJy/sr sec^1/2]
    fov_surf_deg2 = np.pi * (param["fov"].to(u.deg)/2.)**2.
    mapping_speed = fov_surf_deg2/(nefd_spectro_per_beam)**2 
    mapping_speed = mapping_speed.to(u.deg**2 / (u.Jy/u.sr)**2 / u.h) # deg^2 / (Jy/sr)^2 / hour

    # Add to the dictionary
    param["mapping_speed"] = mapping_speed
    param["sigma_array"] = sigma_array
    param["omega_beam"] = beam_area
    param["beam_fwhm_arcsec"] = fwhm_beam.to(u.arcsec)
    param["nefd"] = nefd
    param["nefd_spectro"] = nefd_spectro
    param["sigma_array"] = sigma_array
    param["mapping_speed"] = mapping_speed

    return param

if __name__ == "__main__":

    concerto_param = derive_useful_param(concerto_param)
    
    print("Frequency GHz = ", concerto_param["nu"])
    print("===== NEFD, NEI:")
    print("Sensitivity Dual Band [mJy.s1/2]:",  concerto_param["nefd"])

    print("Sensitivity per spectral element [mJy.s1/2]:", concerto_param["nefd_spectro"])
    print("Sensitivity per spectral element for one whole array [mJy.s1/2]", concerto_param["nefd_spectro"]/np.sqrt(concerto_param["nkids"])) # numbers as in Serra+2016
    
    print("Sigma array [MJy/sr.s^1/2] = ", concerto_param["sigma_array"].value)
    print("===== Mapping Speed:")
    print(
        "Mapping speed [deg^2 / (Jy/sr)^2 / hour] = ",
        concerto_param["mapping_speed"].value * 1.0e15,"[x10^-15]",
    )


    # Compute Point Source sensitivity
    t = compute_PS_sensitivity(concerto_param)
    print("===== Survey sensitivity:")
    print("Sigma Survey [mJy] = ", t[0].to(u.mJy).value)
    print("Sigma LCII [Lsol] = ", t[1].value, "[x10^9]")

    #Compute the Point source sensitivity for SIDES on a finer grid and save it in txt file
    print("==> Compute sensitivity on a finer frequency grid for SIDES...")
    sides_param =  derive_useful_param(sides_param)
    tsides = compute_PS_sensitivity(sides_param)
    print("Sigma Survey [mJy] = ", tsides[0].to(u.mJy).value)

    nu_vec = sides_param['nu'].value
    Sens_vec = tsides[0].to(u.mJy).value
    omega_beam =  sides_param['omega_beam'].value
    txtfile = open('SIDES_sensitivity_1Ghz_res.txt', 'w')
    str = '#col 1: frequency in GHz\n#col2: sensitivity in mJy\n#col3: beam area in sr\n'
    for k in range(0, len(nu_vec)):
        str += '{} {} {}\n'.format(nu_vec[k],Sens_vec[k],omega_beam[k])
    txtfile.write(str)
    txtfile.close()


    #Compute the Point source sensitivity for a 100h 30*30 arcmin^2 field at low resolution
    print("==> Compute sensitivity for a 100h deep pointing at low res...")
    deep_lowres_100h_params =  derive_useful_param(deep_lowres_100h_params)
    tdeep = compute_PS_sensitivity(deep_lowres_100h_params)
    print("Sigma Survey [mJy] = ", tdeep[0].to(u.mJy).value)

    nu_vec = deep_lowres_100h_params['nu'].value
    Sens_vec = tdeep[0].to(u.mJy).value
    omega_beam =  deep_lowres_100h_params['omega_beam'].value
    txtfile = open('100h_30x30_lowres.txt', 'w')
    str = '#col 1: frequency in GHz\n#col2: sensitivity in mJy\n#col3: beam area in sr\n'
    for k in range(0, len(nu_vec)):
        str += '{} {} {}\n'.format(nu_vec[k],Sens_vec[k],omega_beam[k])
    txtfile.write(str)
    txtfile.close()

    embed()

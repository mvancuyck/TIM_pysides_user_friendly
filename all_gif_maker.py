from astropy.io import fits
import matplotlib.pyplot as plt
import imageio
import numpy as np
from matplotlib.colors import LogNorm
from astropy.wcs import WCS
import astropy.units as u
import scipy.constants as cst
import pandas as pd
from scipy.interpolate import interp1d
from IPython import embed
from astropy.coordinates import Angle
import matplotlib.ticker as ticker

def ra_formatter(deg):
    """Format RA ticks in mm:ss."""
    angle = Angle(deg, unit=u.deg)
    total_seconds = angle.to(u.arcmin).value * 60
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def load_noise():
    noise_model_HF = pd.read_csv('tim_sw_loading.tsv', sep='\t')
    noise_model_LF = pd.read_csv('tim_lw_loading.tsv', sep='\t')

    lambda_HF = noise_model_HF["# Wavelength[um]"]*1e3 #nm
    nu_HF = cst.c/(lambda_HF*1e-9)/1e9 #GHz
    nHF = noise_model_HF["NEI[Jy/sr s^1/2]"]
    lambda_LF = noise_model_LF["# Wavelength[um]"]*1e3 #nm
    nu_LF = cst.c/(lambda_LF*1e-9)/1e9 #GHz
    nLF = noise_model_LF["NEI[Jy/sr s^1/2]"]
    freq_noise = np.concatenate((nu_LF[::-1], nu_HF[::-1]))
    noise = (np.concatenate((nLF[::-1], nHF[::-1]))*u.Jy/u.sr).to(u.MJy/u.sr)

    f = interp1d(freq_noise, noise, bounds_error=False, kind='linear', fill_value='extrapolate')
    return f

def generate_white_noise_map(size,sigma):
    """
    Generates a 2D white noise map with values following a Gaussian distribution.
    
    Parameters:
        size (tuple): Dimensions of the map (height, width).
        fwhm (float): Full-width at half maximum (FWHM) of the Gaussian distribution.
        
    Returns:
        np.ndarray: 2D white noise map.
    """
    # Standard deviation from FWHM
    #sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    # Generate white noise with Gaussian distribution
    noise_map = np.random.normal(loc=0, scale=sigma, size=size)
    return noise_map

output_gif = f"gif_frames/a_frequency_sweep_noise.gif"

ciifile =        "OUTPUT_TIM_CUBES_FROM_UCHUU/pySIDES_from_uchuu_TIM_tile0_0.2deg_1deg_res20arcsec_dnu4.0GHz_CII_de_Looze_smoothed_MJy_sr.fits"
fullfile =       "OUTPUT_TIM_CUBES_FROM_UCHUU/pySIDES_from_uchuu_TIM_tile0_0.2deg_1deg_res20arcsec_dnu4.0GHz_full_de_Looze_smoothed_MJy_sr.fits"
alllinesfille =  "OUTPUT_TIM_CUBES_FROM_UCHUU/pySIDES_from_uchuu_TIM_tile0_0.2deg_1deg_res20arcsec_dnu4.0GHz_all_lines_de_Looze_smoothed_MJy_sr.fits"
contfile =       "OUTPUT_TIM_CUBES_FROM_UCHUU/pySIDES_from_uchuu_TIM_tile0_0.2deg_1deg_res20arcsec_dnu4.0GHz_continuum_smoothed_MJy_sr.fits"

fits_list = (fullfile, contfile, ciifile, fullfile,)
vmin_list = (0.1,      0.1,      1e-3,    0.1,  )
vmax_list = (5,        5,        3,       5, )
noisebool_list = (False,False,False, True, True)
label_list = ('CIB+lines','CIB', '[CII]', 'CIB+lines+noise' )

with fits.open(ciifile) as hdul:
    data_cube = hdul[0].data
    header = hdul[0].header

header['CRVAL1'] = +0.1
header['CRVAL2'] = 0.5
wcs = WCS(header, naxis=2) 
# Get frequency axis info
n_freq = header['NAXIS3']
start_freq = header['CRVAL3']  # Start frequency
delta_freq = header['CDELT3']  # Frequency increment

noise = load_noise()

# Frequency values
frequencies = start_freq + np.arange(n_freq) * delta_freq
noise_TIM =noise(frequencies/1e9) *u.MJy/u.sr
freq_CII = 1900.53690000
z_cii = freq_CII / (frequencies/1e9) - 1 

Npix =  (0.2*u.deg**2) / (np.pi * (( (1.22 * cst.c) / (1.9 * frequencies) * u.rad )**2).to(u.deg**2) )
Ndet = np.ones(len(frequencies))*63
w = np.where(frequencies/1e9>943.61)
Ndet[w] = 51
t_int = 200*3600*Ndet/Npix
sigma = (noise_TIM / np.sqrt(t_int)).value


# Prepare GIF frames
frames = []
for i in reversed(range(n_freq)):

    BS = 10; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS); mk = 3; l=1
    fig, axs = plt.subplots(1,4,figsize=(9,6), dpi=200, subplot_kw={'projection':wcs})
    for j, (vmin, vmax, label, noisebool, fits_file) in enumerate(zip(vmin_list, vmax_list, label_list, noisebool_list,fits_list)):

        data_cube = fits.getdata(fits_file)
        mean = np.mean(data_cube, axis=(1,2))
        if(label!='[CII]'):  
            data_cube -= mean[:, np.newaxis, np.newaxis]
            mean = np.zeros(len(mean))

        if(noisebool): noise = generate_white_noise_map(data_cube[i, :, :].shape,sigma[i]+mean[i])
        else: noise = np.zeros(data_cube[i, :, :].shape)
        if(label=='[CII]'): im = axs[j].imshow(data_cube[i, :, :]+vmin+noise, origin='lower', cmap='cividis', norm=LogNorm(vmin=vmin, vmax=vmax), interpolation='None')
        else: im = axs[j].imshow(data_cube[i, :, :]+vmin+noise, origin='lower', cmap='cividis',vmin=-2, vmax=vmax,  interpolation='None')
        cbar = fig.colorbar(im, ax=axs[j], pad=0.2)
        #fig.axes[1].set(xlabel='$\\rm I_{\\nu}$ ['+header.get('BUNIT', '')+']', fontsize=10)
        #if('CIB+lines+' in label): cbar.set_label(f'{label}'+'\n['+header.get('BUNIT', '')+']' , labelpad=-30, y=-0.07, rotation=0)
        cbar.set_label('['+header.get('BUNIT', '')+']' , labelpad=-18, y=-0.06, rotation=0)
        if('noise' in label): axs[j].set_title(label, pad=10)
        else: axs[j].set_title('$\\rm I_{\\nu}$ '+label, pad=10)
        lon = axs[j].coords[0]
        lat = axs[j].coords[1]
        lat.set_major_formatter('d.d')
        lon.set_major_formatter('d.d')
        lon.set_axislabel('RA')
        if(j==0): lat.set_axislabel('Dec')
        else: lat.set_axislabel(' ')
        lat.set_ticks(spacing=15 * u.arcmin)
        lon.set_ticks(spacing=6 * u.arcmin)
    fig.tight_layout()
    fig.suptitle(f"Frequency: {frequencies[i]/1e9:.0f}GHz"+', $\\rm z_{[CII]}$='+f'{z_cii[i]:.2f}')
    fig.subplots_adjust(top=1.1)
    # Save the frame to an image
    temp_filename = f"gif_frames/frame_{i}.png"
    fig.savefig(temp_filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # Append to frames
    frames.append(imageio.imread(temp_filename))

# Create GIF
imageio.mimsave(output_gif, frames, fps=3, loop=0)  # Adjust fps for speed+

print(f"GIF saved to {output_gif}")


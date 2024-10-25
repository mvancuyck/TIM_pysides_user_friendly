import sys
import os
from pysides.load_params import *
import matplotlib
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
from gen_all_sizes_TIM_cubes import sorted_files_by_n
import camb
import glob

#Some parameters
sides = pickle.load( open('/home/mvancuyck/Desktop/comparison_SSS/pySIDES_from_uchuu/dict.p', 'rb'))
TIM_params = load_params('PAR_FILES/Uchuu_cubes_for_TIM.par')

for n, (tile_sizeRA, tile_sizeDEC) in enumerate(TIM_params['tile_sizes']):

    file = f'dict_dir/pySIDES_from_uchuu_{tile_sizeRA}deg_x_{tile_sizeDEC}deg_pks.p'
    dict = pickle.load( open(file, 'rb'))

    nb_subfields = len(dict.keys())

    BS=5; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS); lw=1; mk=1; elw=1; ft=BS
    fig, ax = plt.subplots(figsize=(3,3), sharex=True, dpi=200)
    
    for iz, (z_center,dz,  c) in enumerate(zip(TIM_params['z_centers'], TIM_params['dz'], ('r', 'g', 'b', 'orange'))):

        if(f'dz{dz}' in sides['pks'].keys()):
            if(f'z{z_center}' in sides['pks'][f'dz{dz}'].keys()):
                if(f'{tile_sizeRA}x{tile_sizeDEC}deg2' in sides['pks'][f'dz{dz}'][f'z{z_center}'].keys()):
                    k = sides['pks'][f'dz{dz}'][f'z{z_center}'][f'{tile_sizeRA}x{tile_sizeDEC}deg2']['tile_0']['CII_de_Looze']['k [per_Mpc3]'].value
                    pkmean = sides['pks'][f'dz{dz}'][f'z{z_center}'][f'{tile_sizeRA}x{tile_sizeDEC}deg2']['tile_0']['CII_de_Looze']['P_of_k_mean'][:len(k)]
                    pkstd = sides['pks'][f'dz{dz}'][f'z{z_center}'][f'{tile_sizeRA}x{tile_sizeDEC}deg2']['tile_0']['CII_de_Looze']['P_of_k_std'][:len(k)]
                    ax.loglog(k[1:],pkmean[1:],c='gray',label=f'Other z={z_center}'+'$\\rm \\pm$'+f'{dz}', markersize=mk, lw=lw )
                    ax.fill_between(k[1:],pkmean[1:]-pkstd[1:], pkmean[1:]+pkstd[1:], color='gray',alpha=0.2 )
        
        k = dict['0'][f'pk_3D_z{z_center}_CII_de_Looze']['k_out_sphere #Mpc-1'].value
        pk_sphere_list = []
        for l in range(nb_subfields):
            pk_sphere_list.append(dict[f'{l}'][f'pk_3D_z{z_center}_CII_de_Looze']['pk_out_sphere #Jy2sr-2Mpc3'])
        pk_sphere_list = np.asarray(pk_sphere_list)
        mean = np.mean(pk_sphere_list, axis=0)
        std  = np.std( pk_sphere_list, axis=0) 
        ax.loglog(k[1:],mean[1:],'-|',c=c,label=f'z={z_center}'+'$\\rm \\pm$'+f'{dz}', markersize=mk, lw=lw )
        ax.fill_between(k[1:],mean[1:]-std[1:], mean[1:]+std[1:], color=c,alpha=0.2 )
        
    ax.set_xlabel('k [$\\rm Mpc^{-1}$]')
    ax.legend(fontsize=ft, loc= 'upper right', frameon=False)
    ax.set_ylabel('P(k) [$\\rm Jy^2.sr^{-2}.Mpc^{-1}$]')
    ax.set_title(f'[CII] spherical power spectrum \n in {nb_subfields} fields of {tile_sizeRA}deg by {tile_sizeDEC}deg')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)

    fig.savefig(f'figures/pySIDES_from_uchuu_{tile_sizeRA}deg_x_{tile_sizeDEC}deg_3d_CII_pk.png', transparent=True)
    plt.show()
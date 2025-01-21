#chmod +x me!

#-- I generate all the sub-catalogues for the field size you specified in the .par, from the 117deg2 Uchuu catalogue.
#-- But you can also download them elsewhere. 
python gen_all_sizes_cat.py 
#-- From these subcatalogues, I generate the angular spectral cubes in the TIM bandpass.
python gen_all_sizes_TIM_angular_spectral_cubes.py


'''
#-- Once you generated some angular spectral cubes, I can generate the gif
python all_gif_maker.py


#-- And I generate physical (i.e Mpc) CII cubes as covered by SIDES, and compute their 3D power spectrum
python gen_all_sizes_TIM_cubes.py 
python compute_all_p_of_k.py
#---compute_all_p_of_k.py saves the p(k)s in dictionaries .p, easier to download on a laptop.

#Then plot the results of the measured 3d power spectra with pk_CII_result.ipynb
#--- With pks_sanitycheck.py you can look at each individual 3d power spectra computed in compute_all_p_of_j.py
python pks_sanitycheck.py

#--- the kmodes_in_rectangle_cubes.ipynb explores which k modes in Mpc-1 units are available  
#--- in the 4 redshift bins of a typicall TIM CII cubes. 
'''
You want to use SIDES, you are in the right place. 

1st, download a SIDES catalogue at: https://data.lam.fr/sides/home

I recommand the 2deg2 pySIDES_from_original.fits (3rd option in Download full catalogues) to make your tests. Then you can ask me for bigger fields from SIDES-Uchuu. 

Or you can also cut your own field size using gen_all_size_cat.py

Once you've got your catalogue, you can create a .par in PAR_FILES, based on the Uchuu_cubes_for_TIM.par, to create angular spectral cubes with your choosen parameters. 

To do so, load your catalog in the .fits format, the .par for SIDES and for the cube, and call pysides.make_cube()

example_angular_spectral_map.py summarizes how the make_cube() works.  

This git also contain a way to generate 3D comoving CII cubes and measure their power spectra. 

kmodes_in_rectangle_cubes.ipynb explores which k modes in Mpc-1 units are available.

spt_smg_obs.ipynb contains my first exploration of pointed observations with TIM. 

all_gif_maker.py generates the 4 panels gif, once the TIM cubes have been generated.



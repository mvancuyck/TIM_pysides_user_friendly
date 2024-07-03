#!/bin/bash
#SBATCH --ntasks-per-node=24
#SBATCH --nodes=5
#SBATCH --cpus-per-task=1
#SBATCH -p batch
#module load mpi/openmpi/4.0.1

#chmod +x me!
python gen_all_size_cat.py 
python gen_all_TIM_cubes.py 
python plot_tim_maps
python variance_0.1deg2.py --recompute
python variance_1deg2.py --recompute
python cross_spec_snr_in_tim.py 'outputs_uchuu/'

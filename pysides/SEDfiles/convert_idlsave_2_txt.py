from IPython import embed #for debugging purpose only
import numpy as np
import pickle
from scipy.io import readsav

datams = readsav('SEDs_evolving_ms_v2.save', python_dict = True)

datasb = readsav('SEDs_SB.save', python_dict = True)

sum_lambda_diff = np.sum(datams["lambda"] -  datasb["lambda"])

if sum_lambda_diff > 1.e-3:
    print("Warning! The lambda grid of the 2 IDL files are too different.\nCode stopped!\nNo file generated!!!")
    exit()
else:
    sed_dict = {"lambda": datams["lambda"], "Umean_MS": datams["umean"], "nuLnu_MS_arr": datams["sed_arr"], "Umean_SB": datasb["umean"], "nuLnu_SB_arr": datasb["sed_arr"]}

    #write a pickle dictonnary with the SEDS (fastest to load)
    pickle.dump(sed_dict, open("SED_dict.p", "wb"))

    #write the libraries as a txt file (just to be sure to have them in a simple format in the future)
    for k in range(0, 2):
        if k ==0:
            type = 'MS'
            data = datams
            if k==1:
                type = 'SB'
                data = datasb
            
        file = open('SEDs_'+type+'.txt','w') 
        strout = '# SED library from Magdis+12 used to generate Bethermin model fluxes \n'
        strout += '# This file contain the '+type+' template \n'
        strout += '# First line: 0 and list Umean \n'
        strout += '# After: lambda in microns and nuLnu for the various Umean (normalised to LIR = 1 Lsun) \n'
        strout += '0'
        for u in data["umean"]: strout += ' {:0.12f}'.format(u)
        strout += '\n'

        for k in range(0, np.size(data["lambda"])):
            strout += '{:0.12f}'.format(data["lambda"][k])
            for nuLnu in data["sed_arr"][k,:]:
                strout += ' {:0.12e}'.format(nuLnu)
            strout += '\n'
 
        file.write(strout)  
        file.close()
    

# Load and save the finegrids  

data_finegrid = readsav('SED_U_finegrids.save', python_dict = True)

sed_finegrid_dict = {"Umean": data_finegrid["ugrid"], "dU": data_finegrid["delta_u"], "lambda":  data_finegrid["lambda_grids"], "nuLnu_MS_arr": data_finegrid["sed_u_ms"], "nuLnu_SB_arr": data_finegrid["sed_u_sb"]}

pickle.dump(sed_finegrid_dict, open("SED_finegrid_dict.p", "wb"))

#write the libraries as a txt file (just to be sure to have them in a simple format in the future)
for k in range(0, 2):
    if k ==0:
        type = 'MS'
        sed_arr = data_finegrid["sed_u_ms"]
    if k==1:
        type = 'SB'
        sed_arr = data_finegrid["sed_u_sb"]
            
    file = open('SEDs_finegrid_'+type+'.txt','w') 
    strout = '# Fine-grid (interpolated) SEDs from Magdis+12 used to generate Bethermin model fluxes \n'
    strout += '# This file contain the '+type+' template \n'
    strout += '# First line: 0 and list Umean \n'
    strout += '# After: lambda in microns and nuLnu for the various Umean (normalised to LIR = 1 Lsun) \n'
    strout += '0'
    for u in data_finegrid["ugrid"]: strout += ' {:0.12f}'.format(u)
    strout += '\n'

    for k in range(0, np.size(data_finegrid["lambda_grids"])):
        strout += '{:0.12f}'.format(data_finegrid["lambda_grids"][k])
        for nuLnu in sed_arr[k,:]:
            strout += ' {:0.12e}'.format(nuLnu)
        strout += '\n'
 
    file.write(strout)  
    file.close()

    print("Fine-grid pickle and txt files generated!")

embed()


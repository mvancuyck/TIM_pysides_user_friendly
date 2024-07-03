import numpy as np
import scipy.integrate as spint
import pickle
from IPython import embed

SED_file = 'SED_finegrid_dict.p'

dict_SED = pickle.load(open(SED_file, "rb"))

types = ['MS', 'SB']

dict_ratios = {'Umean': dict_SED['Umean'], "dU": dict_SED['dU']}

for type in types:

    ratio_vec = np.zeros(np.size(dict_SED['Umean']))
    
    for k in range(0, len(dict_SED['Umean'])):

        L_cum = spint.cumtrapz(dict_SED['nuLnu_'+type+'_arr'][k,:], x = np.log(dict_SED['lambda']), initial = 0)
        
        sel = np.where((dict_SED['lambda'] > 8) & (dict_SED['lambda'] < 1000))
        test = spint.simps(dict_SED['nuLnu_'+type+'_arr'][k,sel[0]], x = np.log(dict_SED['lambda'][sel[0]]))
        
        LIR = np.interp([8., 1000.], dict_SED['lambda'], L_cum)
        LIR = LIR[1] - LIR[0]
        
        LFIR = np.interp([40., 400.], dict_SED['lambda'], L_cum)
        LFIR = LFIR[1] - LFIR[0]
        
        ratio_vec[k] = LFIR / LIR

    dict_ratios['LFIR_LIR_ratio_'+type] = ratio_vec

pickle.dump(dict_ratios, open('LFIR_LIR_ratio.p', "wb"))

embed()

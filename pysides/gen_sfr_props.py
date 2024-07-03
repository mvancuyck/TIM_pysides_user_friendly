from time import time
import numpy as np
import pandas as pd
from scipy.special import erf
from IPython import embed

def gen_sfr_props(cat, params):

    tstart = time()

    print('Generate the star-formation properties...')

    print('Draw quenched galaxies...')

    Ngal = len(cat)

    #Draw randomly which galaxies are quenched using the recipe from Bethermin+17
    Mtz = params['Mt0'] + params['alpha1'] * cat["redshift"] + params['alpha2'] * cat["redshift"]**2
    sigmaz =  params['sigma0'] +  params['beta1'] * cat["redshift"] +  params['beta2'] * cat["redshift"]**2
    qfrac0z = params['qfrac0'] * (1.+cat["redshift"])**params['gamma']
    
    Prob_SF = (1.-qfrac0z) * 0.5 * (1. - erf( ( np.log10(cat["Mstar"]) - Mtz) /sigmaz ) )

    Xuni = np.random.rand(Ngal)

    qflag = Xuni > Prob_SF
    
    cat = cat.assign(qflag = qflag)

    #Generate SFR for non-quenched objects

    print('Generate SFRs...')

    index_SF = np.where(qflag == False)

    m_SF = np.array(np.log10(cat["Mstar"][index_SF[0]] * params['Chab2Salp'] / 1.e9 ))
    z_SF = np.array(cat["redshift"][index_SF[0]])
    r = np.log10(1.+z_SF)
    expr = np.maximum(m_SF - params['m1'] - params['a2'] * r, 0.)

    logSFRms_SF = m_SF - params['m0'] + params['a0'] * r - params['a1'] * expr**2 - np.log10(params['Chab2Salp'])

    logSFRms_SF += params['corr_zmean_lowzcorr'] * (params['zmax_lowzcorr'] - np.minimum(z_SF, params['zmax_lowzcorr'])) / (params['zmax_lowzcorr'] - params['zmean_lowzcorr'])

    Psb = params['Psb_hz'] + params['slope_Psb'] * (params['z_Psb_knee'] - np.minimum(z_SF, params['z_Psb_knee']))

    Xuni = np.random.rand(np.size(Psb))

    issb = (Xuni < Psb )

    SFR_SF = 10. ** ( logSFRms_SF + params['sigma_MS'] * np.random.randn(np.size(logSFRms_SF))
                      + params['logx0'] + issb * (params['logBsb'] - params['logx0']) )

    print('Deal with SFR drawn initially above the SFR limit...')

    too_high_SFRs = np.where( SFR_SF > params['SFR_max'])
    while np.size(too_high_SFRs) > 0:
        SFR_SF[too_high_SFRs[0]] = 10. ** ( logSFRms_SF[too_high_SFRs] + params['sigma_MS'] *
                                         np.random.randn(np.size(too_high_SFRs))
                                         + params['logx0'] + issb[too_high_SFRs] * (params['logBsb'] - params['logx0']) )
        
        too_high_SFRs = np.where( SFR_SF > params['SFR_max'])

    print('Store the results...')

    cat = cat.assign(SFR = np.zeros(len(cat)))
    cat = cat.assign(issb = np.zeros(len(cat), dtype = bool))

    cat.loc[index_SF[0], "SFR"] = SFR_SF
    cat.loc[index_SF[0], "issb"] = issb

    tstop = time()

    print(len(cat), 'galaxy SFRs generated in ', tstop-tstart, 's')
    
    return cat

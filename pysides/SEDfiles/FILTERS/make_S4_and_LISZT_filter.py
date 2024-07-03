import numpy as np

#generate square filters for S4 and LISZT

c = 299792458 

filter_names_S4 = ['S4_20GHz', 'S4_27GHz', 'S9_39GHz' , 'S4_93GHz', 'S4_145GHz', 'S4_225GHz', 'S4_278GHz']
numin_S4 = np.array([17.5,24.03,30.03,75.33,124.7,194.635,255.76])
numax_S4 = np.array([22.5,29.97,47.97,110.67,165.3,255.375,300.34])

filter_name_LISZT = ['LISZT500GHz', 'LISZT590GHz', 'LISZT690GHz', 'LISZT815GHz', 'LISZT960GHz', 'LISZT1130GHz', 'LISZT1330GHz', 'LISZT1560GHz', 'LISZT1840GHz', 'LISZT2170GHz','LISZT2550GHz', 'LISZT3000GHz']
numin_LISZT = np.array([400,472,552,652,768,904,1064,1248,1472,1736,2040,2400])
numax_LISZT = np.array([600,708,828,978,1152,1356,1596,1872,2208,2604,3060,3600])



for todo in [zip(filter_names_S4, numin_S4, numax_S4), zip(filter_name_LISZT, numin_LISZT, numax_LISZT)]:
    for filter, numin, numax in todo:
        lambda_min_um = c / numax * 1.e-3
        lambda_max_um = c / numin *1.e-3
        lambda_grid =  lambda_min_um + (lambda_max_um - lambda_min_um) * np.arange(-0.2,1.2,0.02)
        t = np.zeros_like(lambda_grid)
        t[(lambda_grid >= lambda_min_um) & (lambda_grid <= lambda_max_um)] = 1

        file = open(filter+'.dat', 'w')
        txt_out = '#lambda in microns, transmission\n'
        for wout, tout in zip(lambda_grid, t):
            txt_out += '{} {}\n'.format(wout, tout)
        file.write(txt_out)
        file.close()





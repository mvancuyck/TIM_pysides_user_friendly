import pickle
from astropy.table import Table
from astropy.io import fits
import astropy.units as u
from IPython import embed
import os

def intersect(list_a, list_b):
    return list(set(list_a) & set(list_b))

def gen_outputs(cat, params):

    if os.path.exists(params['output_path']) == False:
        print('Create '+params['output_path'])
        os.makedirs(params['output_path'])

    if params['gen_pickle'] == True:
        print('Export the catalog to pickle... (',params['output_path']+params['run_name']+'.p', ')')
        pickle.dump(cat, open(params['output_path']+params['run_name']+'.p', 'wb'))
        
    if params['gen_fits'] == True:
        print('Export the catalog to FITS... (',params['output_path']+params['run_name']+'.fits', ')')
        ap_table = Table.from_pandas(cat)

        #add units to the fields

        col_in = list(cat.columns)

        col_unit = ['Dlum']
        for col in intersect(col_in, col_unit):
            ap_table[col] = ap_table[col] * u.Mpc

        col_unit = ['Mstar', 'Mhalo']
        for col in intersect(col_in, col_unit):
            ap_table[col] = ap_table[col] * u.Msun

        ol_unit = ['SFR']
        for col in intersect(col_in, col_unit):
            ap_table[col] = ap_table[col] * u.Msun / u.Lsun

        col_unit = ['LIR', 'LFIR', 'LCII_Lagache', 'LCII_de_Looze']
        for col in intersect(col_in, col_unit):
            ap_table[col] = ap_table[col] * u.Lsun

        col_unit = ['LprimCO10']
        for col in intersect(col_in, col_unit):
            ap_table[col] = ap_table[col] * u.K * u.km / u.s * u.pc**2

        col_unit = []
        for wl in params['lambda_list']: 
            col_unit.append('S'+str(wl))
        for col in intersect(col_in, col_unit):
            ap_table[col] = ap_table[col] * u.Jy

        col_unit = ['ICII_Lagache', 'ICII_de_Looze', 'ICI10', 'ICI21']
        for k in range(0,20):
            col_unit.append('ICO'+str(k+1)+str(k))
        for col in intersect(col_in, col_unit):
            ap_table[col] = ap_table[col] * u.Jy * u.km / u.s

        fits_filename = params['output_path']+params['run_name']+'.fits'
            
        ap_table.write(fits_filename, overwrite=True)

        print('Add the parameters used for the simulation in the FITS header...')

        hdu = fits.open(fits_filename, mode = 'update')

        for item in list(params.items()):
        
            comment_string = item[0] + ' = ' + str(item[1])

            hdu[1].header['COMMENT'] = comment_string

        hdu.close()

    return True
        

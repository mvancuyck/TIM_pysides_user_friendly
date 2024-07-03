import pandas as pd
import vaex as vx

def load_uchuu_cats(catfile):

    print("Load the catalog generated after performing abundance mathcing using UCHUU simulation, to get RA, Dec, z, Mhalo, and Mstar...")

    cat_v = vx.open(catfile) #load the fits file with vaex
    
    cat = cat_v.to_pandas_df(['redshift', 'ra', 'dec', 'Mhalo', 'Mstar']) #convert it to pandas data frame
    
    return(cat)

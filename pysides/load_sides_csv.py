import pandas as pd

def load_sides_csv(catfile, nrows = None):

    print("Load the catalog CSV generated from the original IDL code to get RA, Dec, z, Mhalo, and Mstar...")

    cat_IDL = pd.read_csv(catfile, sep=',', nrows = nrows)
    
    cat = pd.DataFrame(data = cat_IDL, columns=['redshift', 'ra', 'dec', 'Mhalo', 'Mstar'])
    
    return(cat)

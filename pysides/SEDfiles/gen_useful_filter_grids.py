import generate_filter_grid

path_filters = 'FILTERS/'

#The code can be slow if the filter are very finely sampled: e.g. PACS or Planck
#Do not hesitate to comment filters

#generate_filter_grid.gen_grid(path_filters+'MIPS_024.DAT','MIPS24') 
#generate_filter_grid.gen_grid(path_filters+'pacs_070.txt', 'PACS70') 
#generate_filter_grid.gen_grid(path_filters+'pacs_100.txt', 'PACS100') 
#generate_filter_grid.gen_grid(path_filters+'pacs_160.txt', 'PACS160') 
#generate_filter_grid.gen_grid(path_filters+'spire_PSW_rsrf.txt' , 'SPIRE250')
#generate_filter_grid.gen_grid(path_filters+'spire_PMW_rsrf.txt' , 'SPIRE350')
#generate_filter_grid.gen_grid(path_filters+'spire_PLW_rsrf.txt' , 'SPIRE500')
#generate_filter_grid.gen_grid(path_filters+'LABOCA870.DAT','LABOCA870') 
#generate_filter_grid.gen_grid(path_filters+'NIKA1200.DAT' , 'NIKA1200')
#generate_filter_grid.gen_grid(path_filters+'NIKA2000.DAT' , 'NIKA2000')
#generate_filter_grid.gen_grid(path_filters+'Planck_updated_857.dat', 'Planck857')
#generate_filter_grid.gen_grid(path_filters+'Planck_updated_545.dat', 'Planck545')
#generate_filter_grid.gen_grid(path_filters+'Planck_updated_353.dat', 'Planck353')
#generate_filter_grid.gen_grid(path_filters+'Planck_updated_217.dat', 'Planck217')
#generate_filter_grid.gen_grid(path_filters+'Planck_updated_143.dat', 'Planck143')
#generate_filter_grid.gen_grid(path_filters+'Planck_updated_100.dat', 'Planck100')
#generate_filter_grid.gen_grid(path_filters+'90GHz_SPT.DAT', 'SPT90')
#generate_filter_grid.gen_grid(path_filters+'150GHz_SPT.DAT', 'SPT150')
#generate_filter_grid.gen_grid(path_filters+'220GHz_SPT.DAT', 'SPT220')
#generate_filter_grid.gen_grid(path_filters+'spire_PSW_extended_rsrf.txt', 'SPIRE250_ext')
#generate_filter_grid.gen_grid(path_filters+'spire_PMW_extended_rsrf.txt', 'SPIRE350_ext')
#generate_filter_grid.gen_grid(path_filters+'spire_PLW_extended_rsrf.txt', 'SPIRE500_ext')
generate_filter_grid.gen_grid(path_filters+'SCUBA2_450new.DAT', 'SCUBA2_450')
generate_filter_grid.gen_grid(path_filters+'SCUBA2_850.DAT', 'SCUBA2_850')

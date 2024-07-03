import generate_filter_grid

path_filters = 'FILTERS/'

filter_list = ['S4_20GHz', 'S4_27GHz', 'S9_39GHz' , 'S4_93GHz', 'S4_145GHz', 'S4_225GHz', 'S4_278GHz', 'LISZT500GHz', 'LISZT590GHz', 'LISZT690GHz', 'LISZT815GHz', 'LISZT960GHz', 'LISZT1130GHz', 'LISZT1330GHz', 'LISZT1560GHz', 'LISZT1840GHz', 'LISZT2170GHz','LISZT2550GHz', 'LISZT3000GHz']

for filter in filter_list:
    generate_filter_grid.gen_grid(path_filters+filter+'.dat' , filter)

import generate_filter_grid

path_filters = 'FILTERS/'

filter_list = ['PRIMA_1A_1',
               'PRIMA_1A_2',
               'PRIMA_1A_3',
               'PRIMA_1A_4',
               'PRIMA_1A_5',
               'PRIMA_1A_6',
               'PRIMA_1B_1',
               'PRIMA_1B_2',
               'PRIMA_1B_3',
               'PRIMA_1B_4',
               'PRIMA_1B_5',
               'PRIMA_1B_6',
               'PRIMA_2A',
               'PRIMA_2B',
               'PRIMA_2C',
               'PRIMA_2D']

for filter in filter_list:
    generate_filter_grid.gen_grid(path_filters+filter+'.dat' , filter, renorm_lambda = 1.e-4)

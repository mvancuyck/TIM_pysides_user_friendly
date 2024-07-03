import numpy as np

def print_scientific_notation(number):
    string = f"{number:.1E}"
    int_part = string[:string.find("E")]
    slope = int(string[string.find("E")+1:])
    if(slope == 0): return f"{int_part}", 0
    return int_part, slope #f"{int_part} \\times "+"10^{"+f"{slope}"+"}"

def print_scientific_notation_error(number, error):
    string = f"{number:.1E}"
    int_part = string[:string.find("E")]
    slope = int(string[string.find("E")+1:])
    error_in_same_oom = error / 10**(float(slope))
    int_error = np.round(error_in_same_oom, 1)
    if(slope == 0): return( f"{int_part} \\pm  {int_error}")
    return( f"({int_part} \\pm  {int_error}) \\times "+"10^{"+f"{slope}"+"}")


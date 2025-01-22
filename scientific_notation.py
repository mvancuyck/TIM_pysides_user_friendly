import numpy as np

def scientific_notation(number):
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


def print_scientific_notation(number):
    if(number<1):
        number*=60
        string = f"{number:.1E}"
        int_part = string[:string.find("E")]
        slope = int(string[string.find("E")+1:])
        a = '0.'
        for i in range(abs(slope)-1):    
            a+='0'
        a+=str(int(float(int_part)*10))
        return r'\textbf{'+a+'h}'

        return a
    elif(number>1 and number<10): 

        return r'\textbf{'+str(np.round(number,1))+'h}'
    else: 

        return str(np.round(number).astype(int))+'h' #f"${int_part}"+r" \times "+"10^{"+f"{slope}"+"}$"
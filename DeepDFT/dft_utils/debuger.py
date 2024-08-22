import inspect



def print_structure(var, indent=0):
    # 这个函数是用来调试的,用来查看数据结构  
    if isinstance(var, dict):  
        print(' ' * indent + '{')  
        for key in var:  
            print(' ' * (indent + 2) + 'key:' + key, end=' ')  
            print_structure(var[key], indent + 2)  
        print(' ' * indent + '}')  
    elif isinstance(var, list):  
        print(' ' * indent + '[')  
        for item in var:  
            print_structure(item, indent + 2)  
        print(' ' * indent + ']')  
    else:  
        print(' ' * indent + type(var).__name__)

def structure(var, indent=0):  
    output = ""  
    if isinstance(var, dict):  
        output += ' ' * indent + "{\n"  
        for key in var:  
            output += ' ' * (indent + 2) + "key:" + key + " "  
            output += structure(var[key], indent + 2)  
        output += ' ' * indent + "}\n"  
    elif isinstance(var, list):  
        output += ' ' * indent + "[\n"  
        for item in var:  
            output += structure(item, indent + 2)  
        output += ' ' * indent + "]\n"  
    else:  
        output += ' ' * indent + type(var).__name__ + "\n"  
    return output

def debug_devideline(info:str, dash:str="=", length=40, capital=False):
    dash_half_num = round((length-len(info))/2)
    if dash_half_num <= 0: 
        dash_half_num = 1 
    if capital:
        info = info.upper()
    print(f"{dash*dash_half_num}{info}{dash*dash_half_num}")

'''
def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]
'''

def debug_check_var(var, var_name:str, interest_attrs:list=None, repo:bool=True, devideline:bool=True):
    if devideline:
        debug_devideline(f"check var {var_name}", capital=True)
    print(f"{var_name} Type: {type(var)}")
    if interest_attrs is not None:
        for attr in interest_attrs:
            if hasattr(var, attr):
                print(f"{var_name} {attr}: {getattr(var, attr)}")
    if repo:
        print(f"{var_name} Repo: {var}")
    if devideline:
        debug_devideline(f"check var {var_name} end", capital=True)

def debug_print(topic, info):
    print(f"{topic}: {info}")
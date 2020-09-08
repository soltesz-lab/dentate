import fnmatch 
import yaml
import os 
import pandas   as pd

def make_param_path_str(path):
    l = []
    for x in path:
        l.append(str(x))
    return '.'.join(l)

pattern = '*.yaml'
toplevel = {}  
columns = set([])
files = os.listdir('.') 
for name in files: 
    if fnmatch.fnmatch(name, pattern):
        stream = open(name, 'r')
        data = yaml.safe_load(stream)
        stream.close()
        for k in data:
            result_dict = data[k]
            for i in list(result_dict.keys()):
                params = result_dict[i]
                kvdict = {}
                for p in params:
                    param_path = p[:-1]
                    param_val = p[-1]
                    param_key = make_param_path_str(param_path)
                    param_kv = (param_key, param_val)
                    columns.add(param_key)
                    kvdict[param_key] = param_val
                result_dict[i] = kvdict
            if k in toplevel:
                toplevel[k].update(result_dict)
            else:
                toplevel[k] = result_dict

for k in toplevel:
    data_dict = toplevel[k]
    df_dict = {}
    for i in data_dict:
        param_dict = data_dict[i]
        param_tuple = tuple( param_dict[c] for c in columns )
        df_dict[i] = param_tuple
    print(df_dict)
    df = pd.DataFrame.from_dict(df_dict, orient='index', columns=list(columns))
    df.to_csv('netclamp.optimize.%s.csv' % k)
    print(df)
        

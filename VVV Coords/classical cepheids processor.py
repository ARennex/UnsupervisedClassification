import numpy as np
import pandas as pd

import os,glob
from string import digits

def open_file(path):
    data = pd.read_csv(path,delim_whitespace=True,header=None,dtype='str')
    data['RA'] = data[4].str.cat(data[[5, 6]], sep=' ')
    data['DEC'] = data[7].str.cat(data[[8, 9]], sep=' ')
    return data[['RA','DEC']]

base_path = os.getcwd()
path = base_path+'/classical cepheids.txt'

data = open_file(path)

print(data)

data.to_csv('classical cepheid coords.csv',index=False, header=None)

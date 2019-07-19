import numpy as np
import pandas as pd

import os,glob
from string import digits

from tqdm import tqdm

base_path = os.getcwd()

types_to_process = ['cep','ecl','rrlyr','lpv']
for file_type in types_to_process:
    single_type_path = base_path + '/' + file_type
    contents_path = single_type_path + '/phot/I' + '/OGLE-*.dat'
    print("Loaded files in: " + single_type_path + '/phot/I' + ". Investigating!")

    #Get list of all files of each type
    files1 = np.array(list(glob.iglob(contents_path, recursive=True)))

    processed_path = single_type_path + '/Processed Files/*.dat'

    files2 = np.array(list(glob.iglob(processed_path, recursive=True)))

    print('File Type: ' + file_type)
    print('Original Length: ', len(files1))
    print('Processed Length: ', len(files2))

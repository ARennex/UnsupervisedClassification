import numpy as np
import pandas as pd

import os,glob
from string import digits

base_path = os.getcwd()

def subclass_mode(path):
    subclass_dict = {}
    need a process to read in the file type from the ident.dat files
    read that in first, build a dict of object ids and subtypes
    match that to the main file and use that to save to specific subfolders?

    object_identity = pd.read_csv(path+'ident.dat', header=None, usecols=[0,1], sep=' ')
    print(object_identity)
    keys, values = object_identity[0].values(),object_identity[1].values()
    subclass_dict = dict(zip(keys, values))

    return subclass_dict

path = os.getcwd()
subclass_mode(path)
break

def main_loop():
    processed_objects = 0
    max_objects = 8000

    types_to_process = ['cep','ecl','rrlyr','lpv']
    for file_type in types_to_process:
        processed_objects = 0

        single_type_path = base_path + '/' + file_type
        contents_path = single_type_path + '/phot/I' + '/OGLE-*.dat'
        print("Loaded file: " + single_type_path + '/phot/I' + ". Processing!")

        #These files are already sorted by mjd, no need to do it myself

        #Check if the processed files directory exists
        os.makedirs((base_path + '/' + file_type + '/Laptop Files'), exist_ok=True)

        #Get list of all files of each type
        files1 = np.array(list(glob.iglob(contents_path, recursive=True)))

        print('Files loaded from: ', files1)

        for file in files1: #Added tqdm so I can check the progress
            file_signature = ''.join(c for c in file if c in digits)

            #To prevent this running for far too long, skip after 8000 objects
            if processed_objects >= max_objects:
                print("Over Max Objects")
                break

            with open(file, 'r') as input_file:
                index = 0

                for line in input_file:
                    values = line.split(' ')
                    values = list(filter(None, values))

                    if index == 0:
                        try:
                            previous_value_time = float(values[0])
                            previous_value_mag = float(values[1])
                            index += 1
                        except Exception as e:
                            print(line.split(' '))
                            print(file)
                            exit()

                        processed_objects += 1

                    with open(single_type_path+'/Laptop Files/output_{num:0>3}.dat'.format(num=file_signature), 'a') as output_file:
                        new_line = [str(float(values[0])-previous_value_time),
                                    str(float(values[1])-previous_value_mag),
                                    values[2]]

                        new_line = " ".join(new_line)
                        output_file.write(new_line)

                        previous_value_time = float(values[0])
                        previous_value_mag = float(values[1])

main_loop()

# import shutil
# types_to_process = ['cep','ecl','rrlyr','lpv']
# for file_type in types_to_process:
#     shutil.make_archive('OGLE-'+file_type, 'zip', file_type+'/Laptop Files/')
#     #shutil.make_archive('ATLAS-'+file_type+'.zip', 'zip', file_type+'/Laptop Files/')

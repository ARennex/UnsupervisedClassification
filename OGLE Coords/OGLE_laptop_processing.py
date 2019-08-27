import numpy as np
import pandas as pd

import os,glob
import os.path
from string import digits

from astropy.coordinates import SkyCoord
from astropy import units as u

base_path = os.getcwd()

divide_to_subclasses = True
save_coord_list_for_VVV = True

def ra_dec_conversion(row):
    c = SkyCoord(row[2], row[3], unit=(u.hourangle, u.deg))
    return (c.ra / u.deg).value,(c.dec / u.deg).value
    #return pd.Series((c.ra / u.deg).value,(c.dec / u.deg).value)

def VVV_coords(path,type,input_ogle_data):
    #print(input_ogle_data.apply(ra_dec_conversion, axis=1))
    #ogle_data['ra'],ogle_data['dec'] = input_ogle_data.apply(ra_dec_conversion, axis=1)
    ogle_data = input_ogle_data.apply(ra_dec_conversion, axis=1)
    print(ogle_data)
    ogle_data = ogle_data.str.split(',',expand=True)
    print(ogle_data)
    ogle_data = ogle_data.loc[(ogle_data['ra'] >= 170) & (ogle_data['ra'] <= 281) &
       (ogle_data['dec'] >= -75) & (ogle_data['dec'] <= -20)]

    length = len(ogle_data)
    if length > 10000:
        dfs = np.split(ogle_data, [int(length/2)], axis=0)
        dfs[0].to_csv(path+type+"VVVCoordsPartOne.csv",index=False, header=None)
        dfs[1].to_csv(path+type+"VVVCoordsPartTwo.csv",index=False, header=None)
    else:
        ogle_data.to_csv(path+type+"VVVCoords.csv",index=False, header=None)

def subclass_mode(path):
    subclass_dict = {}
    #need a process to read in the file type from the ident.dat files
    #read that in first, build a dict of object ids and subtypes
    #match that to the main file and use that to save to specific subfolders?
    identity_path = path+'/ident.dat'
    if os.path.isfile(identity_path):
        object_identity = pd.read_csv(identity_path, header=None, usecols=[0,1,2,3,4], delim_whitespace=True)
        if save_coord_list_for_VVV == True:
            for unique_type in pd.unique(object_identity[1]):
                unique_type='ELL'
                print(unique_type)
                VVV_coords(path,unique_type,object_identity[object_identity[1] == unique_type])
                exit()
        keys, values = object_identity[0].tolist(),object_identity[1].tolist()
        subclass_dict = dict(zip(keys, values))
        return subclass_dict
    else:
        return None

path = os.getcwd()
path = path + '/ecl'
result = subclass_mode(path)
exit()

def main_loop():
    processed_objects = 0
    max_objects = 8000

    types_to_process = ['cep','ecl','rrlyr','lpv']
    for file_type in types_to_process:
        processed_objects = 0

        single_type_path = base_path + '/' + file_type
        contents_path = single_type_path + '/phot/I' + '/OGLE-*.dat'
        print("Loaded file: " + single_type_path + '/phot/I' + ". Processing!")

        if divide_to_subclasses == True:
            subclass_mode(single_type_path)

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

import numpy as np
import pandas as pd

import os,glob
import os.path
from string import digits

from astropy.coordinates import SkyCoord
from astropy import units as u

base_path = os.getcwd()

divide_to_subclasses = True
save_coord_list_for_VVV = False
types_to_process = ['cep','ecl','rrlyr','lpv']
types_to_process = ['cep','ecl']

def ra_dec_conversion(row):
    c = SkyCoord(row[0], row[1], unit=(u.hourangle, u.deg))
    return (c.ra / u.deg).value,(c.dec / u.deg).value

def VVV_coords(path,type,input_ogle_data):
    ogle_data = pd.DataFrame([ra_dec_conversion(x[2:4]) for x in input_ogle_data.values.tolist()],columns=['ra','dec'])
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
                print("Processing Subclass: ", unique_type, " for VVV crossmatching.")
                VVV_coords(path,unique_type,object_identity[object_identity[1] == unique_type])
        keys, values = object_identity[0].tolist(),object_identity[1].tolist()
        subclass_dict = dict(zip(keys, values))
        return subclass_dict
    else:
        return None

def main_loop():
    processed_objects = 0
    max_objects = 8000

    for file_type in types_to_process:
        processed_objects = 0

        single_type_path = base_path + '/' + file_type
        contents_path = single_type_path + '/phot/I' + '/OGLE-*.dat'
        print("Loaded file: " + single_type_path + '/phot/I' + ". Processing!")

        #Collect dictionary
        dict_returned_succesful = False
        if divide_to_subclasses == True:
            subclass_dict=subclass_mode(single_type_path)
            if subclass_dict != None:
                dict_returned_succesful = True

        #These files are already sorted by mjd, no need to do it myself

        #Check if the processed files directory exists
        os.makedirs((base_path + '/' + file_type + '/Laptop Files'), exist_ok=True)
        #If so make the sub-driectories for each type
        if dict_returned_succesful == True:
            all_subclasses = set(subclass_dict.values())
            processed_objects = {}
            for subclass in all_subclasses:
                os.makedirs((base_path + '/' + file_type + '/Laptop Files/'+subclass), exist_ok=True)
                processed_objects[subclass] = 0

        #Get list of all files of each type
        files1 = np.array(list(glob.iglob(contents_path, recursive=True)))
        print('Files loaded from: ', files1)

        for file in files1: #Added tqdm so I can check the progress
            _ , ogle_file_identity = os.path.split(file)
            ogle_file_identity = ogle_file_identity[:-4]
            if dict_returned_succesful == True:
                subclass = subclass_dict[ogle_file_identity]+'/'
            else:
                subclass = ''

            #To prevent this running for far too long, skip after 8000 objects
            #This version modified to account for subclasses and non subclasses
            if dict_returned_succesful == True:
                if processed_objects[subclass_dict[ogle_file_identity]] >= max_objects:
                    if all(value > 8000 for value in processed_objects.values()):
                        print("Over Max Objects")
                        break
                    else:
                        continue
            else:
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
                            print("File: ", file, " crashed at split: ",line.split(' '))
                            exit()

                        if dict_returned_succesful == True:
                            processed_objects[subclass_dict[ogle_file_identity]] += 1
                        else:
                            processed_objects += 1

                    #with open(single_type_path+'/Laptop Files/'+subclass+'output_{num:0>3}.dat'.format(num=file_signature), 'a') as output_file:
                    with open(single_type_path+'/Laptop Files/'+subclass+'output_{name}.dat'.format(name=ogle_file_identity), 'a') as output_file:
                        new_line = [str(float(values[0])-previous_value_time),
                                    str(float(values[1])-previous_value_mag),
                                    values[2]]

                        new_line = " ".join(new_line)
                        output_file.write(new_line)

                        previous_value_time = float(values[0])
                        previous_value_mag = float(values[1])

main_loop()

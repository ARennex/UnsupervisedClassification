import numpy as np
import pandas as pd

import os,glob
from string import digits

def open_ogle(path,columns = [0,1,2]):

    if 'cep' in path:
        starting_location = 44
        var_type='cep'
    elif 'ecl' in path:
        starting_location = 26
        var_type='ecl'
    elif 'lpv' in path:
        starting_location = 46
        var_type='lpv'
    elif 'rrlyr' in path:
        starting_location = 45
        var_type='rrlyr'
    elif 'T2CEP' in path:
        starting_location = 43
        var_type='T2CEP'
    else:
        starting_location = 0
        var_type='Error'

    f = open(path, "r")
    ra,dec=[],[]
    for x in f:
        ra.append(x[starting_location:starting_location+10])
        dec.append(x[starting_location+11:starting_location+22])
    return ra,dec,var_type

base_path = os.getcwd() #TODO Go up two steps and put the data there
regular_exp1 = base_path + '/**/ident.dat'
#regular_exp1 = base_path + '/**/phot/I/OGLE-*.dat'

files1 = np.array(list(glob.iglob(regular_exp1, recursive=True)))
print(files1)

coords = []

#files1 = [files1[0],files1[0]]

#for file in files1:
#    ra,dec,var_type = open_ogle(file)
#    print(len(ra), var_type)
#    lists = [ra,dec,[var_type]*len(ra)]
#    paired_coords = [val for tup in zip(*lists) for val in tup]
#    coords.append(paired_coords)
    #break
#flat_list = [item for sublist in coords for item in sublist]
#coords = np.array(flat_list).reshape((-1, 3))
#coords = pd.DataFrame(coords)

#print(coords[0])
#coords = coords.loc[(int(coords[0]) >= 170) & (int(coords[0]) <= 270) &
#    (int(coords[1]) >= -75) & (int(coords[1]) <= -20)]

#coords.to_csv("CoordsWithVarType.csv",index=False, header=None)
#coords.to_csv("CoordsWithoutVarType.csv",index=False, header=None, columns = [0,1])

#RA 183.13410351 degrees
#DEC 	-72.64993034 degrees

#256.48199263 degrees
#Latitude DEC -24.20920040 degrees

#coord_halves = np.split(coords, [int(len(coords)/2)], axis=0)

#coord_halves[0].to_csv("CoordsWithVarType1.csv",index=False, header=None)
#coord_halves[0].to_csv("CoordsWithoutVarType1.csv",index=False, header=None, columns = [0,1])

#coord_halves[1].to_csv("CoordsWithVarType2.csv",index=False, header=None)
#coord_halves[1].to_csv("CoordsWithoutVarType2.csv",index=False, header=None, columns = [0,1])

def single_object_processor():
    written_lines = 0
    max_lines = 1000000

    processed_objects = 0
    max_objects = 30000

    types_to_process = ['cep','ecl','rrlyr','lpv']
    #types_to_process = ['cep']
    for file_type in types_to_process:
        single_type_path = base_path + '/' + file_type
        contents_path = single_type_path + '/phot/I' + '/OGLE-*.dat'
        print("Loaded file: " + single_type_path + '/phot/I' + ". Processing!")

        #These files are already sorted by mjd, no need to do it myself

        #Check if the processed files directory exists
        os.makedirs((base_path + '/' + file_type + '/Processed Files'), exist_ok=True)

        #Get list of all files of each type
        files1 = np.array(list(glob.iglob(contents_path, recursive=True)))
        #print(files1)

        for file in files1:
            file_signature = ''.join(c for c in file if c in digits)
            #print(type(file_signature))

            with open(file, 'r') as input_file:
                index = 0

                for line in input_file:
                    values = line.split(' ')

                    if index == 0:
                        previous_value_time = float(values[0])
                        previous_value_mag = float(values[1])
                        index += 1

                    #To prevent this running for far too long, skip after 50000 objects
                    processed_objects += 1
                    if processed_objects >= max_objects:
                        break

                    #open(single_type_path+'/Processed Files/output_{:08d}.csv'.format(file_signature), 'a')
                    #{num:0>3}
                    with open(single_type_path+'/Processed Files/output_{num:0>3}.dat'.format(num=file_signature), 'a') as output_file:
                        new_line = [str(float(values[0])-previous_value_time),
                                    str(float(values[1])-previous_value_mag),
                                    values[2]]
                        #print(new_line)
                        #print(float(values[0]),previous_value_time,
                        #        float(values[1]),previous_value_mag,values[2])
                        new_line = " ".join(new_line)
                        output_file.write(new_line)

                        written_lines += 1
                        previous_value_time = float(values[0])
                        previous_value_mag = float(values[1])

single_object_processor()

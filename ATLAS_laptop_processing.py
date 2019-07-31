import pandas as pd
import numpy as np

import os

base_path = os.getcwd()

current_id = ''
index = 0
written_lines = 0
max_lines = 1000000

max_objects = 8000

types_to_process = ['cep','rrlyr','lpv','ecl']
for file_type in types_to_process:
    filename = base_path + '/' + file_type + '/' + file_type + 's_aaronb.csv'
    print("Loaded file: " + filename + ". Processing!")

    processed_objects = 0

    #Sort files by date and id
    obs_file = pd.read_csv(filename)
    obs_file = obs_file.sort_values(['objid','mjd'],ascending=True)
    obs_file.to_csv(filename,index=False)

    #Check if the processed files directory exists
    os.makedirs((base_path + '/' + file_type + '/Laptop Files'), exist_ok=True)

    with open(filename, 'r') as input_file:
        for line in input_file:
            values = line.split(',')

            if (current_id != values[0]) or (written_lines > max_lines):
                index += 1
                current_id = values[0]

                try:
                    previous_value_time = float(values[1])
                    previous_value_mag = float(values[3])
                except Exception as e:
                    continue

                #To prevent this running for far too long, skip after 50000 objects
                processed_objects += 1
                if processed_objects >= max_objects:
                    break

            with open(file_type+'/Laptop Files/output_{:08d}.csv'.format(index), 'a') as output_file:
                new_line = [values[0],str(float(values[1])-previous_value_time),
                            values[2],str(float(values[3])-previous_value_mag),
                            values[4],values[5]]
                #print(new_line)
                new_line = ",".join(new_line)
                output_file.write(new_line)

                written_lines += 1
                previous_value_time = float(values[1])
                previous_value_mag = float(values[3])

import zipfile

# def zipdir(path, ziph):
#     # ziph is zipfile handle
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             ziph.write(os.path.join(root, file))
#
# types_to_process = ['cep','ecl','rrlyr','lpv']
# for file_type in types_to_process:
#     zipf = zipfile.ZipFile('ATLAS-'+file_type+'.zip', 'w', zipfile.ZIP_DEFLATED)
#     zipdir('Unsupervised/'+file_type+'Laptop Files/', zipf)
#     zipf.close()

import shutil
types_to_process = ['cep','ecl','rrlyr','lpv']
for file_type in types_to_process:
    shutil.make_archive('ATLAS-'+file_type, 'zip', file_type+'/Laptop Files/')
    #shutil.make_archive('ATLAS-'+file_type+'.zip', 'zip', file_type+'/Laptop Files/')

import pandas as pd
import numpy as np

import os

base_path = os.getcwd()

current_id = ''
index = 0
written_lines = 0
max_lines = 1000000000

max_objects = 8000

types_to_process = ['cep','rrlyr','lpv','ecl']
types_to_process = ['lpv','ecl_close','ecl_distant']
for file_type in types_to_process:
    filename = base_path + '/' + file_type + '_aaronb.csv'
    print("Loaded file: " + filename + ". Processing!")

    processed_objects = 0

    #Sort files by date and id
    obs_file = pd.read_csv(filename)
    obs_file = obs_file.sort_values(['objid','mjd'],ascending=True)
    columns = list(obs_file.columns)
    obs_file.to_csv(filename,index=False)

    output_columns = []
    for searching_column in ['objid','mjd','filter','m','dm']:
        output_columns.append(columns.index(searching_column))
    print(output_columns)

    #Check if the processed files directory exists
    os.makedirs((base_path + '/' + file_type + '/Laptop Files'), exist_ok=True)

    with open(filename, 'r') as input_file:
        for line in input_file:
            values = line.split(',')

            if (current_id != values[output_columns[0]]) or (written_lines > max_lines):
                index += 1
                current_id = values[output_columns[0]]

                try:
                    previous_value_time = float(values[output_columns[1]])
                    previous_value_mag = float(values[output_columns[3]])
                except Exception as e:
                    continue

                #To prevent this running for far too long, skip after 50000 objects
                processed_objects += 1
                if processed_objects >= max_objects:
                    break

            with open(file_type+'/Laptop Files/output_{:s}.csv'.format(current_id), 'a') as output_file:
                new_line = [values[output_columns[0]],str(float(values[output_columns[1]])-previous_value_time),
                            values[output_columns[2]],str(float(values[output_columns[3]])-previous_value_mag),
                            values[output_columns[4]]]
                new_line = ",".join(new_line) + '\n'
                output_file.write(new_line)

                written_lines += 1
                previous_value_time = float(values[output_columns[1]])
                previous_value_mag = float(values[output_columns[3]])

# import shutil
# types_to_process = ['cep','ecl','rrlyr','lpv']
# for file_type in types_to_process:
#     shutil.make_archive('ATLAS-'+file_type, 'zip', file_type+'/Laptop Files/')

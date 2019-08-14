import pandas as pd
import numpy as np

import os

base_path = os.getcwd()

max_objects = 8000 #TODO add in a bit to randomize the 8000

types_to_process = ['cep','rrlyr','lpv','ecl1','ecl2']

types_to_process = ['ecl1','ecl2']
max_objects = 4000
index = 0

for file_type in types_to_process:
    if file_type in ['ecl1','ecl2']:
        trimed_file_type = file_type[0:3]
    else:
        trimed_file_type = file_type
    filename = base_path + '/data_points_' + file_type + '.csv'
    print("Loaded file: " + filename + ". Processing!")

    processed_objects = 0

    #Sort files by date and id
    column_type_dict = {'# sourceID':'Int64','mjd':np.float64,'aperMag3':np.float64,'aperMag3Err':np.float64,'ppErrBits':'Int64','seqNum':'Int64','flag':'Int64'}
    obs_file = pd.read_csv(filename, header=11, dtype=column_type_dict)
    obs_file = obs_file.sort_values(['# sourceID','mjd'],ascending=True)

    obs_file = obs_file.iloc[:800000]

    #Check if the processed files directory exists
    os.makedirs((base_path + '/VVV-' + trimed_file_type), exist_ok=True)

    previous_value_time = 0
    previous_value_mag = 0
    #last_line = a
    #index = 0 #Commenting this out since it was only giving 4000 eclipsing
    current_id = ''
    def batch_processor(a):
        global processed_objects
        global previous_value_time
        global previous_value_mag
        global index
        global current_id
        if processed_objects >= max_objects:
            #print('Over Max')
            return
        if (current_id != a['# sourceID']):
            index += 1
            current_id = a['# sourceID']
            processed_objects += 1
            previous_value_time = a['mjd']
            previous_value_mag = a['aperMag3']
        with open(base_path + '/VVV-' + trimed_file_type+'/output_{:08d}.csv'.format(index), 'a') as output_file:
            new_line = [str(a['# sourceID']),str(a['mjd']-previous_value_time),
                        str(a['aperMag3']-previous_value_mag),
                        str(a['ppErrBits']),str(a['flag']),'\n']
            new_line = ",".join(new_line)
            output_file.write(new_line)

            previous_value_time = a['mjd']
            previous_value_mag = a['aperMag3']

        if index > 3999:
            print(index)
        return

    obs_file.apply(batch_processor, axis = 1)

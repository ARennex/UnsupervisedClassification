import pandas as pd
import numpy as np

import os

base_path = os.getcwd()

max_objects = 8000 #TODO add in a bit to randomize the 8000

types_to_process = ['cep','rrlyr','lpv','ecl1','ecl2']
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
    #obs_file.to_csv(filename,index=False)

    #Check if the processed files directory exists
    os.makedirs((base_path + '/VVV-' + trimed_file_type), exist_ok=True)

    previous_value_time = 0
    previous_value_mag = 0
    #last_line = a
    index = 0
    current_id = ''
    def batch_processor(a):
        global processed_objects
        global previous_value_time
        global previous_value_mag
        global index
        global current_id
        #print(index)
        #print(current_id)
        #print(max_objects)
        if processed_objects >= max_objects:
            print('Over Max')
            return
        if (current_id != a['# sourceID']):
            print('new object')
            index += 1
            current_id = a['# sourceID']
            processed_objects += 1
            previous_value_time = a['mjd']
            previous_value_mag = a['aperMag3']
        with open(base_path + '/VVV-' + trimed_file_type+'/output_{:08d}.csv'.format(index), 'a') as output_file:
            print(str(a['ppErrBits']),str(a['flag']))
            new_line = [str(a['# sourceID']),str(a['mjd']-previous_value_time),
                        str(a['aperMag3']-previous_value_mag),
                        str(a['ppErrBits']),str(a['flag']),'\n']
            new_line = ",".join(new_line)
            output_file.write(new_line)

            previous_value_time = a['mjd']
            previous_value_mag = a['aperMag3']
        return

    obs_file.apply(batch_processor, axis = 1)

    # with open(filename, 'r') as input_file:
    #     for line in input_file:
    #         values = line.split(',')
    #
    #         if (current_id != values[0]) or (written_lines > max_lines):
    #             index += 1
    #             current_id = values[0]
    #
    #             try:
    #                 previous_value_time = float(values[1])
    #                 previous_value_mag = float(values[3])
    #             except Exception as e:
    #                 continue
    #
    #             #To prevent this running for far too long, skip after 50000 objects
    #             processed_objects += 1
    #             if processed_objects >= max_objects:
    #                 break
    #
    #         with open(base_path + '/VVV-' + trimed_file_type+'/output_{:08d}.csv'.format(index), 'a') as output_file:
    #             new_line = [values[0],str(float(values[1])-previous_value_time),
    #                         values[2],str(float(values[3])-previous_value_mag),
    #                         values[4],values[5]]
    #             new_line = ",".join(new_line)
    #             output_file.write(new_line)
    #
    #             written_lines += 1
    #             previous_value_time = float(values[1])
    #             previous_value_mag = float(values[3])

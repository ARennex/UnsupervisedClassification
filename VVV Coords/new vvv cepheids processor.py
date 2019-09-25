import numpy as np
import pandas as pd

import os,glob
from string import digits

base_path = os.getcwd()

index = 0

column_type_dict = {'# sourceID':'Int64','mjd':np.float64,'aperMag3':np.float64,'aperMag3Err':np.float64,'ppErrBits':'Int64','seqNum':'Int64','flag':'Int64','ID':'str','HJD':np.float64,'Ksmag':np.float64,'e_Ksmag':np.float64}

classical_data = pd.read_csv(base_path+'/classicalCepheidsVVVData.csv',skiprows=11,dtype=column_type_dict,sep=',')
print(classical_data)
classical_data=classical_data.sort_values(['# sourceID','mjd'],ascending=True)
valid_rows = np.append(np.arange(0,35),[36,37])
print(valid_rows)
type2_data = pd.read_csv(base_path+'/asu.tsv',skiprows=valid_rows,sep=';')
#type2_data['HJD'] = type2_data['HJD'].replace(' ', '')
#type2_data.astype({'HJD':np.float64}).dtypes
#type2_data = pd.read_csv(base_path+'/asu.tsv',skiprows=valid_rows,dtype=column_type_dict,sep=';',skip_blank_lines=True)
print(type2_data)
type2_data = type2_data.sort_values(['ID','HJD'],ascending=True)

def classical_processor(a):
    global processed_objects
    global previous_value_time
    global previous_value_mag
    global index
    global current_id
    if (current_id != a['# sourceID']):
        current_id = a['# sourceID']
        previous_value_time = a['mjd']
        previous_value_mag = a['aperMag3']
    with open(base_path + '/New VVV Cepheids/output_{:12d}.csv'.format(current_id), 'a') as output_file:
        new_line = [str(a['# sourceID']),str(a['mjd']-previous_value_time),
                    str(a['aperMag3']-previous_value_mag),
                    str(a['ppErrBits']),str(a['flag']),'\n']
        new_line = ",".join(new_line)
        output_file.write(new_line)

        previous_value_time = a['mjd']
        previous_value_mag = a['aperMag3']

    return

def type2_processor(a):
    global processed_objects
    global previous_value_time
    global previous_value_mag
    global index
    global current_id
    if (current_id != a['ID']):
        current_id = a['ID']
        try:
            previous_value_time = float(a['HJD'])
            previous_value_mag = float(a['Ksmag'])
        except:
            print("Point 1")
            print(a)
            exit()
    with open(base_path + '/New VVV Cepheids/output_{:s}.csv'.format(current_id), 'a') as output_file:
        try:
            current_time = float(a['HJD'])
            current_mag = float(a['Ksmag'])
        except:
            print("Point 2")
            print(a)
            exit()

        new_line = [str(a['ID']),str(current_time-previous_value_time),
                    str(current_mag-previous_value_mag),
                    str(a['e_Ksmag']),str(a['e_Ksmag']),'\n']
        new_line = ",".join(new_line)
        output_file.write(new_line)

        try:
            previous_value_time = float(a['HJD'])
            previous_value_mag = float(a['Ksmag'])
        except:
            print("Point 3")
            print(a)
            exit()

    return


#Check if the processed files directory exists
os.makedirs((base_path + '/New VVV Cepheids'), exist_ok=True)

previous_value_time = 0
previous_value_mag = 0
current_id = ''

classical_data.apply(classical_processor, axis = 1)
print(classical_data)

previous_value_time = 0
previous_value_mag = 0
current_id = ''

type2_data.apply(type2_processor, axis = 1)
print(type2_data)

import numpy as np
import pandas as pd

import os,glob
import os.path
from string import digits
import math

"""
After having paired up the ASASSN and VVV objects, the mjd and mag data for each VVV object was downloaded
Each folder corresponds to one variable type and holds multiple star_variables
This will iterate through them and extract each object seperately into its unique file
"""

base_path = os.getcwd()

def vvv_processor(a):
    global processed_objects
    global previous_value_time
    global previous_value_mag
    global current_id
    global current_type
    if (current_id != a['# sourceID']):
        current_id = a['# sourceID']
        previous_value_time = a['mjd']
        previous_value_mag = a['aperMag3']
    with open(current_type + '/output_{:12d}.csv'.format(current_id), 'a') as output_file:
        new_line = [str(a['# sourceID']),str(a['mjd']-previous_value_time),
                    str(a['aperMag3']-previous_value_mag),
                    str(a['ppErrBits']),str(a['flag']),'\n']
        new_line = ",".join(new_line)
        output_file.write(new_line)

        previous_value_time = a['mjd']
        previous_value_mag = a['aperMag3']

    return

types_to_process = ['DSCT','HADS','CWA','CWB','DCEP','DCEPS','RVA','M','EA','EB','EW','ELL','SR','SRD','RRAB','RRC','RRD']
column_type_dict = {'# sourceID':'Int64','mjd':np.float64,'aperMag3':np.float64,'aperMag3Err':np.float64,'ppErrBits':'Int64','seqNum':'Int64','flag':'Int64'}
for types in types_to_process:
    os.makedirs((base_path + '/ASASSN-sourced VVV data/' + types), exist_ok=True)
    file_repository = base_path + '/VVV Matching/' + types
    file_name = glob.glob(file_repository+'/result*.csv')[0] #This is a list, so take the first and only value

    previous_value_time = 0
    previous_value_mag = 0
    current_id = ''
    current_type = base_path + '/ASASSN-sourced VVV data/' + types

    asassn_sourced_vvv_data = pd.read_csv(file_name,skiprows=11,dtype=column_type_dict,sep=',')
    #classical_data=classical_data.sort_values(['# sourceID','mjd'],ascending=True) #Already Sorted!

    asassn_sourced_vvv_data.apply(vvv_processor, axis = 1)
    print(asassn_sourced_vvv_data)

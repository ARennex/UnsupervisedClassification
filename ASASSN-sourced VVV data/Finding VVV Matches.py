import numpy as np
import pandas as pd

import os,glob
import os.path
from string import digits
import math

base_path = os.getcwd()

def subclass_matching(path,types_to_process):
    subclass_dict = {}
    identity_path = path+'/asassn-catalog.csv'
    if os.path.isfile(identity_path):
        object_identity = pd.read_csv(identity_path, usecols=range(0,16))

        object_identity = object_identity.loc[(object_identity['raj2000'] >= 170) & (object_identity['raj2000'] <= 281) &
           (object_identity['dej2000'] >= -75) & (object_identity['dej2000'] <= -20)]

        print(object_identity[['id','raj2000','dej2000']])
        print(object_identity['Type'].unique())

        #object_identity['Type'] = object_identity['Type'].map(lambda x: x.rstrip(':'))

        object_identity = object_identity.loc[object_identity['Type'].isin(types_to_process)]

        object_identity = object_identity[['id','Type','raj2000','dej2000']]
        object_identity.to_csv(path+"/VVV Match Data.csv",index=False, header=None)

        object_identity = object_identity[['raj2000','dej2000']]
        length = len(object_identity)
        if length > 50000:
            rounded_size = math.ceil(length/50000)
            print(rounded_size)
            dfs = np.array_split(object_identity, 3)
            #dfs = np.split(object_identity, [int(length/rounded_size)], axis=0)
            counter = 1
            for df in dfs:
                df.to_csv(path+"/VVV Match Data Coords Only"+str(counter)+".csv",index=False, header=None)
                counter += 1
        else:
            object_identity.to_csv(path+"/VVV Match Data Coords Only.csv",index=False, header=None)
    else:
        print("File Not Found!")

types_to_process = ['DSCT','HADS','CWA','CWB','DCEP','DCEPS','RVA','M','EA','EB','EW','ELL','SR','SRD','RRAB','RRC','RRD']
#subclass_matching(base_path,types_to_process)

def VVV_matches_found(path,types_to_process): #Take in the list of matched coords from VVV and pair them up to find their corresponding variable type
    vvv_object_matches = []
    for counter in [1,2,3]:
        vvv_data_name = "VVV match " + str(counter) +".csv"
        vvv_object_matches.append(pd.read_csv(vvv_data_name, skiprows=14))

    """Merge the seperated vvv object and pair up with the asassn objects"""
    combined_vvv_data = vvv_object_matches[0].append([vvv_object_matches[1],vvv_object_matches[2]])
    print(combined_vvv_data)
    asassn_data_name = "VVV Match Data.csv"
    asassn_object_properties = pd.read_csv(asassn_data_name, header=None, names = ['asassn ID', 'var type', 'upload_RA', 'upload_Dec'])
    print(asassn_object_properties)

    merged_tables = pd.merge(asassn_object_properties,combined_vvv_data,on='upload_RA')
    merged_tables = merged_tables.loc[merged_tables['distance'] > 0.0]
    print("Merged and filtered objects: ", merged_tables)
    print("Counts of each type: ", merged_tables['var type'].value_counts())

    for types in types_to_process:
        os.makedirs((base_path + '/VVV Matching/' + types), exist_ok=True)
        data_to_save = merged_tables.loc[merged_tables['var type'] == types]
        data_to_save.to_csv(base_path + '/VVV Matching/' + types + "/" + types + " Asassn-Vvv Data.csv",index=False)

VVV_matches_found(base_path,types_to_process)

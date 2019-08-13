import numpy as np
import pandas as pd

import os,glob
from string import digits

from astropy.coordinates import SkyCoord
from astropy import units as u

def open_ogle(path,columns = [0,1,2]):
    if 'CEP' in path:
        starting_location = 43
        var_type='CEP'
    elif 'ECL' in path:
        starting_location = 25
        var_type='ECL'
    elif 'LPV' in path:
        starting_location = 45
        var_type='LPV'
    elif 'RRLYR' in path:
        starting_location = 44
        var_type='RRLYR'
    elif 'T2CEP' in path:
        starting_location = 42
        var_type='T2CEP'
    else:
        starting_location = 0
        var_type='Error'

    f = open(path, "r")
    ra,dec=[],[]
    for x in f:
        temp_ra = x[starting_location:starting_location+11]
        temp_dec = x[starting_location+11:starting_location+23]
        c = SkyCoord(temp_ra, temp_dec, unit=(u.hourangle, u.deg))
        temp_ra = (c.ra / u.deg).value
        temp_dec = (c.dec / u.deg).value
        ra.append(temp_ra)
        dec.append(temp_dec)
    #return ra,dec,var_type
    return ra,dec

def open_atlas(path):
    atlas_data = pd.read_csv(path)
    return atlas_data[['ra','dec']]

base_path = os.getcwd() #TODO Go up two steps and put the data there

star_variables = ['CEP','RRLYR','LPV','ECL']
star_variables = ['ECL']
for var_type in star_variables:
    path=base_path + '/' + var_type + 'RADec_aaronb.csv'
    coords = open_atlas(path)
    #print("atlas coords: ",coords['ra'].values, coords['dec'].values)
    path = base_path + '/' + var_type + '_ident.dat'
    ra,dec = open_ogle(path)
    #print("ogle coords: ",ra,dec)
    ogle_data = pd.DataFrame(list(zip(ra,dec)), columns=['ra','dec'])
    combined_coords = coords.append(ogle_data)
    combined_coords = combined_coords.loc[(combined_coords['ra'] >= 170) & (combined_coords['ra'] <= 281) &
       (combined_coords['dec'] >= -75) & (combined_coords['dec'] <= -20)]
    #combined_coords.to_csv(var_type+"Coords.csv",index=False, header=None)

    length = len(combined_coords)
    print(length)
    dfs = np.split(combined_coords, [int(length/2)], axis=0)

    dfs[0].to_csv(var_type+"CoordsPartOne.csv",index=False, header=None)
    dfs[1].to_csv(var_type+"CoordsPartTwo.csv",index=False, header=None)










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

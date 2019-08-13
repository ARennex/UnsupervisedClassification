import numpy as np
import pandas as pd

import os,glob


base_path = os.getcwd()
regular_exp1 = base_path + '/results_*.csv'
files1 = np.array(list(glob.iglob(regular_exp1, recursive=True)))
print(files1)

for file in files1:
    vvv_data = pd.read_csv(file, header=14)
    vvv_data = vvv_data[vvv_data['sourceID'] != 0]
    valuable_data = vvv_data[['sourceID','framesetID','RA','Dec']]
    print(valuable_data)
    valuable_data.to_csv(os.path.splitext(file)[0] + "_vvv_input.csv", index=False)

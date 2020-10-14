import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from numba import jit
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xmltodict
import numpy.polynomial as p
from multiprocessing import Pool
import time
from datetime import datetime , date, timedelta

from fottech_lib.market_data.dmds import DMDSServices
from fottech_lib import instrumentservice
from fottech_lib.market_data.repo import Repo

import project.market_data.repocurves as repoc
from project.market_data.repocurves import RepoCurves


#Loading the indices
file_path = './data/universe_indices.npy'
universe_indices = np.load(file_path)


def get_repo_schedules(universe_indices_ric,business_date):
    dictionary = {}
    for ric in universe_indices_ric:
        print('############################## Index {} ##############################'.format(ric))
        try:
            div_paths = 'RepoCurve/official/{}/PARIS/INTRADAY/equity/{}/sophis'.format(business_date,ric)
            ds = DMDSServices('prod', 'APAC')
            docs = ds.get_documents(div_paths)
            d_s = docs['documents']['document'][0].__values__.get('content')
            repo_schedule = xmltodict.parse(d_s)
            date = repo_schedule['RepoCurve']['@businessDate']
            df = pd.DataFrame(repo_schedule['RepoCurve']['repo'])
            df['#text'] = df['#text'].astype(str)
            df['@term'] = df['@term'].astype(str)

            for i in range(df.shape[0]):
                f_date = datetime.strptime(date, "%Y-%m-%d").date()
                l_date = datetime.strptime(df['@term'][i], "%Y-%m-%d").date()
                delta = l_date - f_date
                if (delta.days >= 0):
                    df['@term'][i] = delta.days
                else:
                    df = df.drop(i, axis = 0)
            df = df.reset_index(drop=True)
            df = df.get_values()
            col1 = df[:,0].tolist()
            col2 = df[:,1].tolist()
            col = [col1 , col2, date]
            dictionary[ric]=col
        except:
            dictionary[ric]=None
    return dictionary



def save_dict(dictionary):
    file_path = './output/universe_repo_processed.json'
    try:
        with open(file_path, 'w') as fp:
            json.dump(dictionary, fp)
        print('file saved')
    except:
        print('For some reasons, the file couldnt be saved')


universe_indices_ric = []
B_to_R = instrumentservice.InstrumentService('prod','APAC')
for index in universe_indices:
    index_ric = B_to_R.transcode(index, target='reuter', partial_match=False)
    if(index_ric != None):
        ric = index_ric[1:]
        universe_indices_ric.append(ric)
dictionary = get_repo_schedules(universe_indices_ric,'latest')
save_dict(dictionary)


path_to_data_Universe = './output/universe_repo_processed.json'

path_to_cleaned_data_Universe = './output/universe_repo_cleaned.json'


print('################## Cleaning dividends for Universe index ##################')

new_dict = {}
with open(path_to_data_Universe) as json_file:
    dictionary = json.load(json_file)

for key in list(dictionary.keys()):
    if (dictionary[key] != None):
        if np.sum(np.isnan(dictionary[key][0])) == 0 and np.sum(np.isnan(list(map(float, dictionary[key][1])))) == 0:
            dictionary[key][1] = list(map(float, dictionary[key][1]))
            new_dict[key] = dictionary[key]

xvals = [90, 180, 365, 730, 1095, 1460, 1825, 2190, 2555, 2920, 3285, 3650, 4015, 4380]
for key in new_dict.keys():
    x = new_dict[key][0]
    y = new_dict[key][1]
    yinterp = np.interp(xvals, x, y)
    # computing new interpolated values
    new_dict[key][0] = xvals
    new_dict[key][1] = yinterp.tolist()

with open(path_to_cleaned_data_Universe, 'w') as fp:
    json.dump(new_dict, fp)
print('file saved')


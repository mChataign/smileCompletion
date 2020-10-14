import os
import json
import numpy as np
import time
import datetime
from datetime import datetime , date, timedelta
import sys
import pandas as pd
import numpy as np
#import xmltodict
import sys
import numpy as np
from numba import jit
import json
import matplotlib.pyplot as plt
import seaborn as sns
#import xmltodict
import numpy.polynomial as p
from multiprocessing import Pool

# FOTTECH
from fottech_lib.market_data import dividends
from fottech_lib.market_data.dividends import Dividends
from fottech_lib.market_data.dmds import DMDSServices
from fottech_lib import instrumentservice

from project import config
import project.market_data.repocurves as repoc
from project.market_data.repocurves import RepoCurves

dmds_date_format = dividends.dmds_date_format
dmds_path_date_format = dividends.dmds_path_date_format


def preprocessed_data_autoencoder_repo(training_rate = 0.8):
    
    #Loading data
    path_to_cleaned_data_SX5E = config.path_to_cleaned_data_SX5E
    path_to_cleaned_data_NKY = config.path_to_cleaned_data_NKY
    path_to_cleaned_data_SPX = config.path_to_cleaned_data_SPX
    path_to_cleaned_data_UKX = config.path_to_cleaned_data_UKX

    with open(path_to_cleaned_data_SX5E) as json_file:
        dictionary_SX5E = json.load(json_file)
    with open(path_to_cleaned_data_NKY) as json_file:
        dictionary_NKY = json.load(json_file)
    with open(path_to_cleaned_data_SPX) as json_file:
        dictionary_SPX = json.load(json_file)
    with open(path_to_cleaned_data_UKX) as json_file:
        dictionary_UKX = json.load(json_file)
    
    # Interpolating the data
    xvals = [90, 180, 365, 730, 1095, 1460, 1825, 2190, 2555, 2920, 3285, 3650, 4015, 4380]
    input_vector_SX5E = []
    input_vector_NKY = [] 
    input_vector_SPX = []
    input_vector_UKX = []

    for key in dictionary_SX5E.keys():
        x = dictionary_SX5E[key][0]
        y = dictionary_SX5E[key][1]
        yinterp = np.interp(xvals, x, y)
        #computing new interpolated values
        dictionary_SX5E[key][0] = xvals
        dictionary_SX5E[key][1] = yinterp
        input_vector_SX5E.append(yinterp.tolist())

    for key in dictionary_NKY.keys():
        x = dictionary_NKY[key][0]
        y = dictionary_NKY[key][1]
        yinterp = np.interp(xvals, x, y)
        #computing new interpolated values
        dictionary_NKY[key][0] = xvals
        dictionary_NKY[key][1] = yinterp
        input_vector_NKY.append(yinterp.tolist())

    for key in dictionary_SPX.keys():
        x = dictionary_SPX[key][0]
        y = dictionary_SPX[key][1]
        yinterp = np.interp(xvals, x, y)
        #computing new interpolated values
        dictionary_SPX[key][0] = xvals
        dictionary_SPX[key][1] = yinterp
        input_vector_SPX.append(yinterp.tolist())

    for key in dictionary_UKX.keys():
        x = dictionary_UKX[key][0]
        y = dictionary_UKX[key][1]
        yinterp = np.interp(xvals, x, y)
        #computing new interpolated values
        dictionary_UKX[key][0] = xvals
        dictionary_UKX[key][1] = yinterp
        input_vector_UKX.append(yinterp.tolist())
    
    # Computing the input_training_set and input_validation_set
    stop_NKY = int(len(input_vector_NKY)*training_rate)
    stop_SPX = int(len(input_vector_SPX)*training_rate)
    stop_SX5E = int(len(input_vector_SX5E)*training_rate)
    stop_UKX= int(len(input_vector_UKX)*training_rate)

    input_training_set_NKY = input_vector_NKY[:stop_NKY]
    input_validation_set_NKY = input_vector_NKY[stop_NKY:]
    input_training_set_SPX = input_vector_SPX[:stop_SPX]
    input_validation_set_SPX = input_vector_SPX[stop_SPX:]
    input_training_set_SX5E = input_vector_SX5E[:stop_SX5E]
    input_validation_set_SX5E = input_vector_SX5E[stop_SX5E:]
    input_training_set_UKX = input_vector_UKX[:stop_UKX]
    input_validation_set_UKX = input_vector_UKX[stop_UKX:]

    input_training_set = np.vstack((input_training_set_NKY,input_training_set_SPX,input_training_set_SX5E,input_training_set_UKX))
    input_validation_set = np.vstack((input_validation_set_NKY,input_validation_set_SPX,input_validation_set_SX5E,input_validation_set_UKX))
    return input_training_set, input_validation_set

def get_repo_schedules_universe(universe_indices_ric,business_date):
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
    
def cleaning_data_universe(path_to_processed_data,path_to_cleaned_data):
	print('################## Cleaning Repos for Universe indices ##################')

	new_dict = {}
	with open(path_to_processed_data) as json_file:
		dictionary = json.load(json_file)

	for key in list(dictionary.keys()):
		if (dictionary[key]!=None):
			if np.sum(np.isnan(dictionary[key][0]))==0 and np.sum(np.isnan(list(map(float,dictionary[key][1]))))==0 :
				dictionary[key][1] = list(map(float,dictionary[key][1]))
				new_dict[key] = dictionary[key]

	xvals = [90, 180, 365, 730, 1095, 1460, 1825, 2190, 2555, 2920, 3285, 3650, 4015, 4380]
	for key in new_dict.keys():
		x = new_dict[key][0]
		y = new_dict[key][1]
		yinterp = np.interp(xvals, x, y)
		#computing new interpolated values
		new_dict[key][0] = xvals
		new_dict[key][1] = yinterp.tolist()
				
	with open(path_to_cleaned_data, 'w') as fp:
		json.dump(new_dict, fp)
	print('file saved')
        
def cleaning_data_universe_Jiang(path_to_processed_data,path_to_cleaned_data):
	print('################## Cleaning Repos for Universe indices Jiang ##################')

	new_dict = {}
	with open(path_to_processed_data) as json_file:
		dictionary = json.load(json_file)

	for key in list(dictionary.keys()):
		if (dictionary[key]!=None):
			if np.sum(np.isnan(dictionary[key][0]))==0 and np.sum(np.isnan(list(map(float,dictionary[key][1]))))==0 :
				dictionary[key][1] = list(map(float,dictionary[key][1]))
				new_dict[key] = dictionary[key]
		
	with open(path_to_cleaned_data, 'w') as fp:
		json.dump(new_dict, fp)
	print('file saved')    
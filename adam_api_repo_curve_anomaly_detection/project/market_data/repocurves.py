import sys
import pandas as pd
from fottech_lib.market_data import dividends
from fottech_lib.market_data.dividends import Dividends
import date_utils
import numpy as np
from numba import jit
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fottech_lib.market_data.dmds import DMDSServices
import xmltodict
import numpy.polynomial as p
from multiprocessing import Pool
import time
import json
from datetime import datetime , date 

dmds_date_format = dividends.dmds_date_format
dmds_path_date_format = dividends.dmds_path_date_format


class RepoCurves:
	''' '''

	def __init__(self,ric,business_date):
		self.ric = ric
		self.business_date = business_date


def ric_to_symbol(ric):

	if(ric=='SPX'):
		return 'SPX'
	elif(ric=='UKX'):
		return 'FTSE'
	elif(ric=='NKY'):
		return 'N225'
	elif(ric=='SX5E'):
		return 'STOXX50E'
	else:
		return ric

def ric_to_id_number(ric):

	if(ric=='SPX'):
		return '1012688'
	elif(ric=='UKX'):
		return '1012676'
	elif(ric=='NKY'):
		return '1012696'
	elif(ric=='SX5E'):
		return '1012679'
	else:
		return ric
  

	
	print("loading spot prices")
	start = time.time()
	ric = ric_to_symbol(ric) 
	paths = ['EquityPrice/official/{}/LONDON/CLOSE/equityIndex/_{}/DTP'.format(day,ric) for day in dates]
	ds = DMDSServices('prod', 'APAC')
	strings = [ds.get_documents(paths)['documents']['document'][i].__values__.get('content') for i in range(len(paths))]
	spot_prices = []
	for string in strings:
		try:
			spot_prices.append(dict(dict(xmltodict.parse(string))['EquityPrice'])['@last_trade_price'])
		except:
			spot_prices.append(None)
	end = time.time()
	print("spot prices loaded")
	print("time to proceed :",end-start,"s")
	return spot_prices

def business_dates(past='20180101'):
	
	day_past = 0
	tenor = '0d'
	date = date_utils.add_tenor(tenor=tenor)
	business_date = date.strftime(dmds_path_date_format)
	bd = [business_date]
	while(business_date >= past):
		day_past -= 1
		tenor = str(day_past) + 'd'
		date = date_utils.add_tenor(tenor=tenor)
		business_date = date.strftime(dmds_path_date_format)
		if(business_date not in bd):
			bd.append(business_date)
	return list(reversed(bd[:-1]))

def get_repo_schedules(ric,dates):
    dictionary = {}
    for business_date in dates:
        try:
            div_paths = 'RepoCurve/official/{}/PARIS/INTRADAY/equity/{}/sophis'.format(business_date,ric)
            ds = DMDSServices('prod', 'APAC')
            docs = ds.get_documents(div_paths)
            d_s = docs['documents']['document'][0].__values__.get('content')
            div_schedule = xmltodict.parse(d_s)
            df = pd.DataFrame(div_schedule['RepoCurve']['repo'])
            df['#text'] = df['#text'].astype(str)
            df['@term'] = df['@term'].astype(str)

            for i in range(df.shape[0]):
                f_date = datetime.strptime(business_date, "%Y%m%d").date()                
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
            col = [col1 , col2]
            dictionary[business_date]=col
        except:
            print('')
    return dictionary

def save_dict(dictionary, ric, past, future):
	file_path = 'data/processed/{}/Repo/repo_schedule_{}_from_{}_to_{}.json'.format(ric,past,future)
	try:
		with open(file_path, 'w') as fp:
			json.dump(dictionary, fp)
		print('file saved')
	except:
		print('For some reasons, the file couldnt be saved')
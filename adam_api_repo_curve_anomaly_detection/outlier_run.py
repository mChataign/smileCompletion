from flask import Flask, render_template, request, jsonify, Markup, url_for
from wtforms import Form, validators
import sqlite3
import os 
import numpy as np
import json
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import pandas as pd
from datetime import datetime , date, timedelta
from time import gmtime, strftime
from fottech_lib import instrumentservice
import project.data_preprocessing as dp
import script_sql.sqlite_populate as ssp
import torch
import torch.nn as nn
import torch.nn.functional as F

# scriptDirectory = os.path.join(os.path.dirname(__file__))
scriptDirectory = 'c:/git/repo'

#Preparing the model
class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()

        self.linear1 = nn.Linear(5, 20)
        self.linear2 = nn.Linear(20, 20)
        self.linear3 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.tanh(x)
        x = self.linear2(x)
        x = F.tanh(x)
        x = self.linear3(x)
        return x
    
ae = AE()
ae.load_state_dict(torch.load(os.path.join(scriptDirectory,'model_objects/jiang_state_dict_model.pth')))

#Loading the indices
file_path = os.path.join(scriptDirectory,'data/universe_indices.npy')
universe_indices = np.load(file_path)

#Datetime
time_zone = strftime("%z", gmtime())
today = datetime.today()
today_time_exact = today.strftime("%d-%b-%Y %H:%M:%S")

#Preparing processed data
universe_indices_ric = []

B_to_R = instrumentservice.InstrumentService('prod','APAC')
for index in universe_indices:
    index_ric = B_to_R.transcode(index, target='reuter', partial_match=False)

    if(index_ric != None):
        ric = index_ric[1:]
        universe_indices_ric.append(ric)

dictionary = dp.get_repo_schedules_universe(universe_indices_ric,'latest')

file_path = os.path.join(scriptDirectory,'output/universe_repo_processed.json')
try:
    with open(file_path, 'w') as fp:
        json.dump(dictionary, fp)
    print('file saved')
except:
    print('For some reasons, the file couldnt be saved')

#cleaning the data
path_to_processed_data = os.path.join(scriptDirectory,'output/universe_repo_processed.json')
path_to_cleaned_data = os.path.join(scriptDirectory,'output/universe_repo_cleaned.json')
dp.cleaning_data_universe_Jiang(path_to_processed_data,path_to_cleaned_data)


#Loading the database
with open(path_to_cleaned_data) as json_file:
	dictionary_cleaned = json.load(json_file)

#Prediction
indices_repo = list(dictionary_cleaned.keys())
indices_ric = ['.'+_ for _ in indices_repo]
indices_dt = [B_to_R.transcode(_, target='description', partial_match=False) for _ in indices_ric]
indices_mnemo = [B_to_R.transcode(_, target='bloomberg', partial_match=False) for _ in indices_ric]

time_series = np.array(list(dictionary_cleaned.values()))[:,0]
repo = np.array(list(dictionary_cleaned.values()))[:,1]
dates_repo = np.array(list(dictionary_cleaned.values()))[:,2]
dates_repo = [datetime(int(_[:4]),int(_[5:7]),int(_[8:])).strftime("%d-%b-%Y") for _ in dates_repo]

class Attrib_test(nn.Module):
    
    def __init__(self):
        
        super (Attrib_test, self).__init__()
        self.attrib = nn.Parameter(torch.zeros([len(indices_repo), 4]), requires_grad = True)
    def forward(self, x):
        return self.attrib

attrib_test = Attrib_test()

optimizer = torch.optim.Adam(list(attrib_test.parameters()), lr=1e-1)

for k in range(len(indices_repo)):        
    res = torch.Tensor([0.])
    # for _ in range(200):
    for _ in range(100):
        res_ = []
        for x,y in zip(time_series[k],repo[k]):
            if x>90:
                res_.append((ae(torch.cat([torch.Tensor([x/252.0]),attrib_test.attrib[k,:]]))[0] - y)**2)
        res = torch.mean(torch.stack(res_))

        optimizer.zero_grad()
        res.backward()
        optimizer.step()

    if k % 50 ==0:
        print(k,'/',len(indices_repo))


repo_decoded =[]        
for k in range(len(indices_repo)):
    pred = []
    for x in time_series[k] :
        if x>=90:
            pred.append((ae(torch.cat([torch.Tensor([x/252.0]),attrib_test.attrib[k,:]])).detach().numpy())[0])
    repo_decoded.append(pred)        

#RMSES
rmses_repo = []
max_error = []
for i in range(len(repo)):
    try:
        repo_cleaned = np.array(repo[i])
        repo_cleaned = repo_cleaned[np.array(time_series[i])>=90]
        # rmses_repo.append(np.sqrt(mean_squared_error(repo[i] ,repo_decoded[i])/len(time_series[i])))
        rmses_repo.append(min(np.sqrt(mean_squared_error(repo_cleaned, repo_decoded[i])),.2))
        max_error.append(min(np.max(np.abs(repo_cleaned - np.array(repo_decoded[i])))/(max(np.max(repo_cleaned)-np.min(repo_cleaned),0.01)),1))
    except:
        rmses_repo.append(0.)
        max_error.append(0.)




        
#And now we round the values
repo_decoded = [[float(_) for _ in inner] for inner in repo_decoded]
rmses_repo = np.array(rmses_repo)
# rmses_repo = np.round_(rmses_repo, decimals=3)
#rmses_forwards = [_**2 for _ in rmses_forwards]
 
# seuil = 0.1
# outliers_indices = [_ for _ in range(len(indices_dt)) if(rmses_repo[_] > seuil)]
# # outliers_indices = [_ for _ in range(len(indices_dt)) if(max_error[_] > seuil)]
# MSE_outliers = rmses_repo[outliers_indices]
# accurate_indices = [_ for _ in range(len(indices_dt)) if(_ not in outliers_indices)]
# MSE_accurate = rmses_repo[accurate_indices]

if len(indices_dt) > 20:
    #KMeans to determine the threshold
    df = pd.DataFrame(rmses_repo)
    kmeans = KMeans(n_clusters=2).fit(df)
    outliers_indices = np.where(kmeans.labels_==0)[0]
    if(len(outliers_indices) >= (len(kmeans.labels_)-len(outliers_indices))):
        outliers_indices = np.where(kmeans.labels_==1)[0]
    MSE_outliers = rmses_repo[outliers_indices]
    accurate_indices = [_ for _ in range(len(indices_dt)) if(_ not in outliers_indices)]
    MSE_accurate = rmses_repo[accurate_indices]
    seuil = (np.max(MSE_accurate) + np.min(MSE_outliers)) /2
else:
    seuil = 0.1
    outliers_indices = [_ for _ in range(len(indices_dt)) if(rmses_repo[_] > seuil)]
    accurate_indices = [_ for _ in range(len(indices_dt)) if(_ not in outliers_indices)]

# ==========================================================================================
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template("template.html",
        universe_indices=universe_indices,
        len_universe_indices=len(universe_indices),
        indices_repo=indices_dt,
        len_indices_repo=len(indices_dt),
        indices_mnemo=indices_mnemo,
        indices_ric=indices_ric,
        repo=repo,
        repo_json=json.dumps(repo.tolist()),
        repo_decoded=repo_decoded,
        repo_decoded_json=json.dumps(repo_decoded),
        rmses_repo=rmses_repo,
        rmses_repo_json=json.dumps(rmses_repo.tolist()),
        labels=json.dumps(time_series.tolist()),
        outliers_indices=outliers_indices,
        len_outliers_indices=len(outliers_indices),
        accurate_indices=accurate_indices,
        len_accurate_indices=len(accurate_indices),
        seuil=seuil,
        time_zone=time_zone,
        today_time_exact=today_time_exact,
        dates_repo=dates_repo)


@app.route('/stop', methods=['GET', 'POST'])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'

app.run(host='0.0.0.0',  port=8000, threaded=True)
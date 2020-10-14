from flask import Flask, render_template, request, jsonify, Markup, url_for
from wtforms import Form, validators
from keras.models import load_model
from keras.models import model_from_json
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

scriptDirectory = os.path.join(os.path.dirname(__file__))
#scriptDirectory = 'C:/Users/45036224/Desktop/Adam_API_Repo/final_api_repo_curve_anomaly_detection'

#Preparing the model
with open(os.path.join(scriptDirectory,'model_objects/autoencoder_CV.json'),'r') as f:
    autoencoder_json = json.load(f)

autoencoder = model_from_json(autoencoder_json)
autoencoder.load_weights(os.path.join(scriptDirectory,'model_objects/autoencoder_CV.h5'))

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
dp.cleaning_data_universe(path_to_processed_data,path_to_cleaned_data)

with open(path_to_cleaned_data) as json_file:
	dictionary_cleaned = json.load(json_file)

#saving the data as sqlite
try:
    os.remove(os.path.join(scriptDirectory, 'output', 'repo.sqlite'))
except:
    print('Aucun Fichier sql avant')

ssp.run_sqlite(scriptDirectory)

#Loading the database
db = os.path.join(scriptDirectory, 'output', 'repo.sqlite')

def normalize_data(repo_schedule):

    min_ = np.min(repo_schedule)
    max_ = np.max(repo_schedule)
    return (repo_schedule - min_) / (max_ - min_)

def predict(repo_schedule):

    return autoencoder.predict(repo_schedule.reshape((1,len(repo_schedule))))

def get_data_repo(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    divs = c.execute("SELECT * FROM repo_schedules").fetchall()
    conn.commit()
    conn.close()
    return np.array(divs)

    
labels = [90, 180, 365, 730, 1095, 1460, 1825, 2190, 2555, 2920, 3285, 3650, 4015, 4380]
def duplicate(testList, n):
    return testList*n
labels = duplicate([labels],146)

#Forwards
data_repo = get_data_repo(db)
indices_repo = data_repo[:,0]
indices_ric = ['.'+_ for _ in indices_repo]
indices_dt = [B_to_R.transcode(_, target='description', partial_match=False) for _ in indices_ric]
indices_mnemo = [B_to_R.transcode(_, target='bloomberg', partial_match=False) for _ in indices_ric]

repo = np.array(data_repo[:,1:-1],dtype=float)
dates_repo = np.array(data_repo[:,-1])
dates_repo = [datetime(int(_[:4]),int(_[5:7]),int(_[8:])).strftime("%d-%b-%Y") for _ in dates_repo]
min_ = np.min(repo,axis=1)
max_ = np.max(repo,axis=1)
min_ = min_.reshape(len(min_),1)
max_ = max_.reshape(len(min_),1)
for i in range(len(min_)):
    if(min_[i]==max_[i]):
        max_[i] = np.array([repo[i][0]])
        min_[i] = np.array([0.])
        if(max_[i]==0):
            max_[i] = np.array([1.])

#I need 3 info divs, divs decoded and corresponded error
repo_normalized = (repo - min_) / (max_ - min_)
for i in range(len(min_)):
    if(np.sum(repo_normalized[i])==0):
        repo_normalized[i] = np.ones((14))

repo_decoded_normalized = autoencoder.predict(repo_normalized)
repo_decoded = repo_decoded_normalized * (max_ - min_) + min_

rmses_repo = []

for i in range(len(repo)):
    try:
        rmses_repo.append(np.sqrt(mean_squared_error(repo_normalized[i] ,repo_decoded_normalized[i])))
    except:
        rmses_repo.append(0.)
        
#And now we round the values
repo = np.round_(repo, decimals=3)
repo_decoded = np.round_(repo_decoded, decimals=3)
rmses_repo = np.round_(rmses_repo, decimals=3)
#rmses_forwards = [_**2 for _ in rmses_forwards]

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
        repo_decoded_json=json.dumps(repo_decoded.tolist()),
        rmses_repo=rmses_repo,
        rmses_repo_json=json.dumps(rmses_repo.tolist()),
        labels=json.dumps(labels),
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

app.run(host='0.0.0.0',  port=7000, threaded=True)

#==============================================================================
# app = Flask(__name__)
# 
# @app.route("/", methods=['GET','POST'])
# def main():
#     if(False):
#         return 0
#     return render_template("controlform.html", indices=indices_dt, len_indices=len(indices_dt), indices_ric=indices_repo)
# 
# @app.route("/register")
# def register():
#     return render_template("register.html")
# 
# @app.route("/login")
# def login():
#     return render_template("login.html")
# 
# @app.route("/about")
# def about():
#     return render_template("about.html")
# 
# #testing line plot
# @app.route("/line", methods=['GET','POST'])
# def line(): 
#     #get data from controlform
#     rps = {}
#     rps_deco = {}
#     errors = {}
#     index = []
#     if(request.method == 'POST'):
#         rmses = rmses_repo
#         selected_indices = []
#         default_value = '9999'
#         for number,indice in enumerate(indices_dt):
#             val = request.form.get(indice, default_value)
#             if(val != '9999'):
#                 selected_indices.append(number)
#                 index.append(indice)
#                 rps[indice] = repo[number].tolist()
#                 rps_deco[indice] = repo_decoded[number].tolist()
#                 errors[indice] = rmses[number]
#         
#         df = pd.DataFrame(rmses)
#         kmeans = KMeans(n_clusters=2).fit(df)
#         outliers_indices = np.where(kmeans.labels_==0)[0]
#         accurate_indices = np.where(kmeans.labels_==1)[0]
#         
#         if(len(outliers_indices) >= (len(kmeans.labels_)-len(outliers_indices))):
#             outliers_indices = np.where(kmeans.labels_==1)[0]
#             accurate_indices = np.where(kmeans.labels_==0)[0]
#        
#         MSE_outliers = [rmses[_] for _ in outliers_indices]
#         MSE_accurate = [rmses[_] for _ in accurate_indices]
#         seuil = (np.min(MSE_outliers)+np.max(MSE_accurate))/2
#         
#         #Now dumping dictionary
#         return render_template('line_chart.html', indices = index, indices_ric=indices_repo, labels=labels, dividends=json.dumps(rps), dividends_decoded=json.dumps(rps_deco), rmses=json.dumps(errors), selected_indices=selected_indices, dates=dates, seuil=seuil)
#         
#     
#     return render_template("controlform.html", indices = indices_dt, indices_ric=indices_repo)
# 
# @app.route("/stats", methods=['GET','POST'])
# def stats():
#     #correct if the user select correct in the form and vice vesra
#     correct_vs_incorrect = {}
#     type_of_error = {}
#     index  = []
#     err = {}
#     textarea = {}
#     if(request.method == 'POST'):
#         indices = indices_dt
#         rmses = rmses_repo
#         default_value = '9999'
#         for number, indice in enumerate(indices):
#             #canva = request.form.get("chart"+indice, default_value)
#             radiobutton = request.form.get("radiobut"+indice, default_value)
#             selectbutton = request.form.get("ErrorOptions"+indice,default_value)
#             text = request.form.get("comment"+indice,default_value)
#             #Now we do stuff only if canva exist:
#             if(selectbutton != '9999'):
#                 correct_vs_incorrect[indice] = radiobutton
#                 #normal_vs_anormal[indice] = None
#                 type_of_error[indice] = selectbutton
#                 index.append(indice)
#                 err[indice] = rmses[number]
#                 textarea[indice] = text
#         
#         df = pd.DataFrame(rmses)
#         kmeans = KMeans(n_clusters=2).fit(df)
#         outliers_indices = np.where(kmeans.labels_==0)[0]
#         accurate_indices = np.where(kmeans.labels_==1)[0]
#         
#         if(len(outliers_indices) >= (len(kmeans.labels_)-len(outliers_indices))):
#             outliers_indices = np.where(kmeans.labels_==1)[0]
#             accurate_indices = np.where(kmeans.labels_==0)[0]
#        
#         MSE_outliers = [rmses[_] for _ in outliers_indices]
#         MSE_accurate = [rmses[_] for _ in accurate_indices]
#         seuil = (np.min(MSE_outliers)+np.max(MSE_accurate))/2
#         
#         P_M = 0
#         for _ in err:
#             if(err[_]<seuil):
#                 P_M += 1
#         N_M = len(err) - P_M
#         TP = 0
#         TN = 0
#         error_distri = {}
#         error_distri["missing values"] = 0
#         error_distri["not smooth"] = 0
#         error_distri["discontinuity"] = 0
#         error_distri["other"] = 0
#         for _ in correct_vs_incorrect:
#             if correct_vs_incorrect[_] == 'correct':
#                if err[_] <= seuil:
#                     TP += 1
#                else:
#                     TN += 1
#                     error_ = type_of_error[_]
#                     error_distri[error_] += 1
#  
#         FP = P_M - TP
#         FN = N_M - TN
# 
#         max_error = max(list(error_distri.values()))
#         dr = 0
#         far = 0
#         if(TP!=0):
#             dr = np.round(TP/(TP+FN), decimals=3)
#         if(FP!=0):
#             far = np.round(FP/(FP+TN), decimals=3)
#         return render_template("statistics.html",text = textarea, dr=dr, far=far, indices=index, indices_ric=indices_repo, correct_vs_incorrect=correct_vs_incorrect, type_of_error=type_of_error, error_distri=list(error_distri.values()),number=len(index),err=err, TP=TP, FP=FP, FN=FN, TN=TN, max_error=max_error)
#     
#     return render_template("statistics.html")
# 
# 
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', threaded=True)
#     #app.run(debug=True)
#==============================================================================

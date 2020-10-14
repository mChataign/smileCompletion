import json
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from live_plotter import live_plotter

import torch

dates = np.empty(0)
repos = np.empty(0)

f_list = [
    '../adam_api_repo_curve_anomaly_detection/data/cleaned/NKY/Repo/repo_schedule_NKY_from_20130101_to_20190726.json',
    '../adam_api_repo_curve_anomaly_detection/data/cleaned/SPX/Repo/repo_schedule_SPX_from_20130101_to_20190726.json',
    '../adam_api_repo_curve_anomaly_detection/data/cleaned/SX5E/Repo/repo_schedule_SX5E_from_20130101_to_20190726.json',
    '../adam_api_repo_curve_anomaly_detection/data/cleaned/UKX/Repo/repo_schedule_UKX_from_20130101_to_20190726.json',
]

for f_ in f_list:
    with open(f_, 'r') as f:
        dictionary = json.load(f)

    new_dict = {}
    for key in list(dictionary.keys()):
        if (dictionary[key] != None):
            if np.sum(np.isnan(dictionary[key][0])) == 0 and np.sum(
                    np.isnan(list(map(float, dictionary[key][1])))) == 0:
                dictionary[key][1] = list(map(float, dictionary[key][1]))
                new_dict[key] = dictionary[key]

    indices = list(new_dict.keys())
    dates = np.hstack([dates, np.array(list(new_dict.values()))[:, 0]])
    repos = np.hstack([repos, np.array(list(new_dict.values()))[:, 1]])

# for date, repo in zip(dates, repos):
#     #     print(len(date),len(repo))
#     plt.plot(np.array(date)[(np.array(date)>=90) & (np.array(date)<=4380)], np.array(repo)[(np.array(date)>=90) & (np.array(date)<=4380)])
#
# plt.show()

import torch.nn as nn
import torch.nn.functional as F


class Functional_encoder(nn.Module):
    def __init__(self):
        super(Functional_encoder, self).__init__()

        self.linear1 = nn.Linear(5, 20)
        self.linear2 = nn.Linear(20, 20)
        self.linear3 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        x = torch.tanh(x)
        x = self.linear3(x)
        return x


class Code(nn.Module):
    def __init__(self):
        super(Code, self).__init__()
        self.code = nn.Parameter(torch.zeros([2645, 4]), requires_grad=True)

    def forward(self, x):
        return self.attrib


n = len(dates)
fe = Functional_encoder()
code = Code()

optimizer_ae = torch.optim.Adam(fe.parameters(), lr=1e-2)
optimizer_code = torch.optim.Adam(code.parameters(), lr=1e-2)

x_vec = np.linspace(0, 1, 1000 + 1)[0:-1]
y_vec = np.zeros(len(x_vec))
line1 = []

for _ in range(1000):

    res = torch.Tensor([0.])
    for __ in range(50):
        k = np.random.choice(n)
        res_ = []
        for x, y in zip(dates[k], repos[k]):
            if x > 90:
                res_.append((fe(torch.cat([torch.Tensor([x / 252.0]), code.code[k, :]]))[0] - y) ** 2)
        res += torch.mean(torch.stack(res_))

    # res += torch.mean(torch.mean(attrib.attrib))**2 + (torch.mean(torch.mean(attrib.attrib**2))-1)**2
    if _ < 25:
        optimizer_ae.zero_grad()
        res.backward()
        optimizer_ae.step()

    else:
        optimizer_ae.zero_grad()
        optimizer_code.zero_grad()
        res.backward()
        optimizer_ae.step()
        optimizer_code.step()

    print(_, np.sqrt(res.item()))
    y_vec[-1] = np.sqrt(res.item())
    line1 = live_plotter(x_vec, y_vec, line1)
    y_vec = np.append(y_vec[1:], 0.0)

for k in range(20):
    date = dates[k]
    repo = repos[k]
    pred = []
    for x in date:
        pred.append((fe(torch.cat([torch.Tensor([x / 252.0]), code.code[k, :]])).detach().numpy())[0])
    plt.figure()
    plt.plot(date, pred)
    plt.plot(date, repo, 'o')
    plt.title(indices[k])
    plt.show()

# saving model
# torch.save(ae.state_dict(),
#            'model_objects/jiang_state_dict_model.pth')
# torch.save(ae,
#            'model_objects/jiang_full_model.pth')

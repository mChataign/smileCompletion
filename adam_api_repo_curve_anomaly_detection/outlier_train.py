import json
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

import torch

dates = np.empty(0)
repos = np.empty(0)

f_list = [
    'data/cleaned/NKY/Repo/repo_schedule_NKY_from_20130101_to_20190726.json',
    'data/cleaned/SPX/Repo/repo_schedule_SPX_from_20130101_to_20190726.json',
    'data/cleaned/SX5E/Repo/repo_schedule_SX5E_from_20130101_to_20190726.json',
    'data/cleaned/UKX/Repo/repo_schedule_UKX_from_20130101_to_20190726.json',
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

for date, repo in zip(dates, repos):
    #     print(len(date),len(repo))
    plt.plot(np.array(date)[(np.array(date)>=90) & (np.array(date)<=4380)], np.array(repo)[(np.array(date)>=90) & (np.array(date)<=4380)])

plt.show()

import torch.nn as nn
import torch.nn.functional as F


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


class Attrib(nn.Module):
    def __init__(self):
        super(Attrib, self).__init__()
        self.attrib = nn.Parameter(torch.zeros([2645, 4]), requires_grad=True)

    def forward(self, x):
        return self.attrib


n = len(dates)
ae = AE()
attrib = Attrib()

optimizer = torch.optim.Adam(list(ae.parameters()) + list(attrib.parameters()), lr=1e-2)

for _ in range(10000):

    res = torch.Tensor([0.])
    for __ in range(50):
        k = np.random.choice(n)
        res_ = []
        for x, y in zip(dates[k], repos[k]):
            if x > 90:
                res_.append((ae(torch.cat([torch.Tensor([x / 252.0]), attrib.attrib[k, :]]))[0] - y) ** 2)
        res += torch.mean(torch.stack(res_))

    res += torch.mean(torch.mean(attrib.attrib))**2 + (torch.mean(torch.mean(attrib.attrib**2))-1)**2
    optimizer.zero_grad()
    res.backward()
    optimizer.step()
    if _ % 100 == 0:
        print(res.detach().numpy()[0])

for k in range(20):
    date = dates[k]
    repo = repos[k]
    pred = []
    for x in date:
        pred.append((ae(torch.cat([torch.Tensor([x / 252.0]), attrib.attrib[k, :]])).detach().numpy())[0])
    plt.figure()
    plt.plot(date, pred)
    plt.plot(date, repo, 'o')
    plt.title(indices[k])
    plt.show()

# saving model
torch.save(ae.state_dict(),
           'model_objects/jiang_state_dict_model.pth')
torch.save(ae,
           'model_objects/jiang_full_model.pth')

import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from live_plotter import live_plotter
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


test = pickle.load(open('NKY_clean', 'rb'))

n_obs = len(test)
n_factor = 15


class Functional_encoder(nn.Module):
    def __init__(self):
        super(Functional_encoder, self).__init__()

        self.linear1 = nn.Linear(n_factor + 2, 20)
        self.linear2 = nn.Linear(20, 1)
        # self.linear2 = nn.Linear(20, 20)
        # self.linear3 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        # x = torch.tanh(x)
        # x = self.linear3(x)
        return x


class Code(nn.Module):
    def __init__(self):
        super(Code, self).__init__()
        # self.code = nn.Parameter(torch.normal(0, 1, size=(n_obs, n_factor)), requires_grad=True)
        self.code = nn.Parameter(torch.zeros((n_obs, n_factor)), requires_grad=True)



fe = Functional_encoder()
code = Code()
optimizer = torch.optim.Adam(fe.parameters(), lr=1e-2)
optimizer_code = torch.optim.Adam(code.parameters(), lr=1e-2)

strikes = torch.log(torch.Tensor([test[0].iloc[0].iloc[4:].index]).T)

x_vec = np.linspace(0, 1, 1000 + 1)[0:-1]
y_vec = np.zeros(len(x_vec))

line1 = []

for __ in range(100000):
    res_ = []
    for k in np.random.choice(n_obs, 200):
        for j in range(len(test[k])):
        # for j in np.random.choice(len(test[k]), 10):
        # filt  = test[k].nBizDays >= 10
            res_.append(torch.mean(
                (fe(torch.cat((strikes, torch.full((21, 1), np.log(test[k].iloc[j].nBizDays / 252.0)),
                               code.code[k, :].repeat(21, 1)), dim=1))[:, 0] - \
                 torch.tensor(test[k].iloc[j].iloc[4:])) ** 2
            ))
    res = torch.mean(torch.stack(res_))

    # res += torch.mean(torch.mean(code.code))**2 + (torch.mean(torch.mean(code.code**2)-1))**2

    if __ < 300:
        optimizer.zero_grad()
        res.backward()
        optimizer.step()

    else:
        optimizer.zero_grad()
        optimizer_code.zero_grad()
        res.backward()
        optimizer.step()
        optimizer_code.step()

    print(__, np.sqrt(res.item()))
    y_vec[-1] = np.sqrt(res.item())
    line1 = live_plotter(x_vec, y_vec, line1)
    y_vec = np.append(y_vec[1:], 0.0)

torch.save(fe.state_dict(), 'fe')
torch.save(code.state_dict(), 'code')

# fe.load_state_dict(torch.load('truc'))

k = np.random.choice(n_obs)
x = np.array(test[0].iloc[0].iloc[5:].index, dtype=float)
y = np.array(test[k].nBizDays, dtype=float)

x, y = np.meshgrid(x,y)

z = np.array(test[k].iloc[:,5:], dtype=float)

u = fe(torch.cat([torch.tensor(np.log(x).reshape((np.prod(x.shape),1))).float(), torch.tensor(np.log(y / 252.0).reshape((np.prod(x.shape),1))).float(),
                               code.code[k, :].repeat(np.prod(x.shape), 1)], dim=1)).detach().numpy().reshape(x.shape)

fig = plt.figure()
ax = plt.axes(projection='3d')


ax.plot_surface(np.log(x), np.log(y), u, cmap='viridis', edgecolor='none', alpha = 0.5)
ax.plot_surface(np.log(x), np.log(y), z, edgecolor='none', alpha = 0.5)

ax.set_title(str(k))
plt.show()


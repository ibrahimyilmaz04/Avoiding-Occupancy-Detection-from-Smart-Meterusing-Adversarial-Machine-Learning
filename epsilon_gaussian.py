CourseFolder=r""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import csv
import os

# hyperparameters
home_number = 5
season_name = "winter"


output_size = 2
hidden_size = 150
layer_size = 2

batch_size = 10000
learning_rate = 0.001

epochs = 200

use_cuda=True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

print('home_'+ str(home_number).zfill(2) + '_' + season_name)
# home_01_sum = pd.read_csv('occupancy_dataset/home2/02_summer.csv')
# home_01_win = pd.read_csv('occupancy_dataset/home2/02_winter.csv')

#home_01 = [home_01_sum, home_01_win]
#data_01 = pd.concat(home_01)

# !pip install patool
# import patoolib
# patoolib.extract_archive(CourseFolder + "occupancy_dataset.rar", outdir=CourseFolder)

data_01 = pd.read_csv(CourseFolder+'occupancy_dataset/home' + str(home_number) + '/' + str(home_number).zfill(2) + '_' + season_name + '.csv')


# data_01.head(600)

def occ_date(c):
    val = c[3:6]
    if val == 'Jul':
        m = '07'
    elif val == 'Aug':
        m = '08'
    elif val == 'Sep':
        m = '09'
    elif val == 'Oct':
        m = '10'
    elif val == 'Nov':
        m = '11'
    elif val == 'Dec':
        m = '12'
    elif val == 'Jan':
        m = '01'
    elif val == 'Feb':
        m = '02'
    elif val == 'Mar':
        m = '03'
    elif val == 'Apr':
        m = '04'
    elif val == 'May':
        m = '05'
    elif val == 'Jun':
        m = '06'

    return c[7:11] + m + c[:2]

occ_dates= []
for i in range(data_01.iloc[:,0].shape[0]):
    occ_dates.append(occ_date(data_01.iloc[i,0]))

plugs_01 = os.listdir(CourseFolder+'occupancy_dataset/home' + str(home_number) + '/plugs/')
plugs_readings_01 = pd.DataFrame()
target_01 = pd.DataFrame()
flage = 1

input_size = len(plugs_01)
#loop plugs directory(01, 02, ... , 12)
for plug in plugs_01:

    readings_01 = pd.DataFrame()
    p = os.listdir(CourseFolder+'occupancy_dataset/home' + str(home_number) + '/plugs/' + plug)

    #loop plugs's subdirectory(ex: 01)
    for f in p:
        is_common = 1
        for plg in plugs_01:
            is_common *= os.path.exists(CourseFolder+'occupancy_dataset/home' + str(home_number) + '/plugs/' + plg + '/' + f)

        if is_common:
            for i in range(len(occ_dates)):
                #file f is in occ_dates
                if occ_dates[i][:] == f[:4] + f[5:7] + f[8:10]:
                    if flage == 1:
                        target_01 = pd.concat([target_01, data_01.iloc[i, 1:]], axis=0)

                    readings_01 = pd.concat([readings_01, pd.read_csv(CourseFolder+'occupancy_dataset/home' + str(home_number) + '/plugs/' + plug + '/' + f, header=None)], sort=False)
    flage = 0

    plugs_readings_01 = pd.concat([plugs_readings_01, readings_01], axis=1)

# plugs_readings_01

# Overall Coverage Percentage

count = plugs_readings_01.count()
#Coverage_Percentage = (sum(plugs_readings_01.count()[0]) - sum(plugs_readings_01.eq(-1).sum())) / sum(plugs_readings_01.count()[0])
# print('Coverage Percentage = ', round(Coverage_Percentage, 4) * 100, '%')

plugs_readings_01.eq(-1).sum()

def drop_missisng(X, y):

    X = np.array(X)
    y = np.array(y)
    Xout = []
    yout = []

    val = -1
    for i in range(0,X.shape[0]-1):
        flag = 1
        for v in X[i]:
            if(v==val):
               flag = 0
        if flag:
           Xout.append(X[i])
           yout.append(y[i])
    return (np.array(Xout),np.array(yout))

X = plugs_readings_01.values.tolist()
y = target_01.values.tolist()

X,Y = drop_missisng(X,y)

#for i in range (11):
#    print (np.count_nonzero(X[:,i]))

# X = np.delete(X,4,1)
# X = np.delete(X,4,1)
# X = np.delete(X,4,1)
# X.shape

# X = X.sum(axis = 1).reshape(-1, 1) #[200000:250000]
#yx = y.sum(axis = 1) #[200000:250000]

#Xx = X[:1000000]
#Yx = Y[:1000000]
#y[i,:]

seq_length = 30

dataX = []
dataY = []

for i in range(0, len(X) - seq_length):
    _x = X[i:i + seq_length]
    _y = Y[i + seq_length]
    dataX.append(_x)
    dataY.append(_y)

X = np.array(dataX)
Y = np.array(dataY)

import torch
torch.cuda.is_available()
# True

# X.reshape(-1, 30)

Y.reshape(-1)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import sklearn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.init as init

# %matplotlib inline

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1)
_, X_test, _, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1)

# Scaling data
#scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
#X_train = scaling.transform(X_train)
#X_test = scaling.transform(X_test)


#scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

# x = torch.FloatTensor(X_train.tolist()).to(device)
# y = torch.LongTensor(y_train.tolist()).to(device)
# train = torch.utils.data.TensorDataset(x, y.reshape(-1))
# train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
#y = y.squeeze_()

x_test = torch.FloatTensor(X_test.tolist()).to(device)
y_test = torch.LongTensor(y_test.tolist()).to(device)
test = torch.utils.data.TensorDataset(x_test, y_test.reshape(-1))
print(np.shape(x_test))

batch_size = 10
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

#y_test = y_test.squeeze_()

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


class RNNClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, sequence_length, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1
        self.sequence_length = sequence_length

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          bidirectional=bidirectional, num_layers=self.n_layers, batch_first=True)
        for param in self.gru.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

        self.fc = nn.Linear(hidden_size, output_size)
        init.xavier_normal_(self.fc.weight.data)
        init.normal_(self.fc.bias.data)

    def forward(self, input, training=True):
        # Note: we run this all at once (over the whole input sequence)

        # input = B x S . size(0) = B
        batch_size = input.size(0)

        # Make a hidden
        hidden = self._init_hidden(batch_size)

        input.view(batch_size, self.sequence_length, self.input_size)

        output, hidden = self.gru(input, hidden)

        # Use the last layer output as FC's input
        # No need to unpack, since we are going to use hidden
        hidden = F.dropout(hidden[-1], training=training)
        fc_output = self.fc(hidden)

        return fc_output

    def _init_hidden(self, batch_size):
        device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size)
        return Variable(hidden).to(device)


# model = nn.Sequential(nn.Linear(input_size, hidden_size),
#                      nn.LeakyReLU(),
#                      nn.Linear(hidden_size, hidden_size),
#                      nn.LeakyReLU(),
#                      nn.Linear(hidden_size, hidden_size),
#                      nn.LeakyReLU(),
#                      nn.Linear(hidden_size, hidden_size),
#                      nn.LeakyReLU(),
#                      nn.Linear(hidden_size, output_size),
#                      nn.Sigmoid())

model = RNNClassifier(input_size, hidden_size, output_size, seq_length,layer_size, False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model.to(device)
print(model)

if use_cuda and torch.cuda.is_available():
    model.cuda()


loss_func = nn.CrossEntropyLoss()

pretrained_model = CourseFolder + 'model_' + str(home_number) + '_' + season_name
print(pretrained_model)
if os.path.exists(pretrained_model):
    print("model exists")
    device = torch.device("cuda")
    model.load_state_dict(torch.load(pretrained_model, map_location="cuda:0"))
    model.to(device)


perturbation_list_gaussian = []
perturbation_list_epsilon = []

def gaussian_epsilon(epsilon):
    mu, sigma = 0, 7.5
    add_noise_gaussian = True
    perturbed_data_full_gaussian = torch.empty(0)
    perturbed_data_full_gaussian = perturbed_data_full_gaussian.to(device)
    global perturbation_list_gaussian

    out = []
    out = np.array(out)
    vmax = x_test.max()
    vmin = x_test.min()
    correct = 0
    num_samples = 0
    i = 0
    flag = True
    perturbed_data_full_epsilon = torch.empty(0)
    perturbed_data_full_epsilon = perturbed_data_full_epsilon.to(device)
    global perturbation_list_epsilon

    x_test_full = torch.empty(0)
    x_test_full = x_test_full.to(device)

    for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
        if batch_idx == 83:
            break

        x_test_var = Variable(X_batch).to(device)
        y_test_var = Variable(y_batch).to(device)
        x_test_var_flatten = torch.flatten(x_test_var)

        if add_noise_gaussian:
            if batch_idx==0:
                noise_gaussian = np.random.normal(mu, sigma, x_test_var_flatten.shape)
                noise_gaussian = torch.from_numpy(noise_gaussian).float().to(device)
            if x_test_var_flatten.shape != noise_gaussian.shape:
                noise_gaussian = np.random.normal(mu, sigma, x_test_var_flatten.shape)
                noise_gaussian = torch.from_numpy(noise_gaussian).float().to(device)
            perturbation_list_gaussian = perturbation_list_gaussian + noise_gaussian.tolist()
            x_test_perturbed_gaussian = noise_gaussian + x_test_var_flatten
        perturbed_data_full_gaussian = torch.cat([perturbed_data_full_gaussian, x_test_perturbed_gaussian], dim=0)
        
        if flag == True:
            x_test_var.requires_grad = True
            net_out = model(x_test_var, False)
            loss = loss_func(net_out, y_test_var)
            model.zero_grad()
            loss.backward()
            data_grad = x_test_var.grad.data

            # fgsm attack
            sign_data_grad = data_grad.sign()
            pertubation_epsilon = epsilon * sign_data_grad * (vmax - vmin)
            perturbation_list_epsilon = perturbation_list_epsilon + torch.flatten(pertubation_epsilon).tolist()
            if batch_idx == 0 :
                perturbed_data_epsilon = x_test_var_flatten
            else:
                perturbed_data_epsilon = x_test_var_flatten + torch.flatten(pertubation_epsilon)
                flag = False
        else:
            perturbed_data_epsilon = x_test_var_flatten - torch.flatten(pertubation_epsilon)
            flag = True
        perturbed_data_full_epsilon = torch.cat([perturbed_data_full_epsilon, perturbed_data_epsilon], dim=0)

        x_test_full = torch.cat([x_test_full, x_test_var_flatten], dim=0)
    return perturbed_data_full_gaussian, perturbed_data_full_epsilon, x_test_full

# eps = 0.1
# eps = 0.01
# eps = 0.001
# eps = 0.0001
# eps = 0.00001

eps = 0.00001

perturbed_data_full_gaussian, perturbed_data_full_epsilon, x_test_full = gaussian_epsilon(eps)

print(perturbed_data_full_gaussian.shape)
print(perturbed_data_full_epsilon.shape)
print(x_test_full.shape)


readings = pd.DataFrame(torch.flatten(perturbed_data_full_gaussian).tolist(), columns=['consump_perturbed_guassian'])
readings['consump_perturbed_epsilon'] = torch.flatten(perturbed_data_full_epsilon).tolist()
readings['consump'] = torch.flatten(x_test_full).tolist()

time_stamp = pd.read_csv(CourseFolder+'occupancy_dataset/home' + str(home_number) + '/' + str(home_number).zfill(2) + '_' + season_name + '.csv')
time_stamp = pd.DataFrame(time_stamp.columns.values.tolist()[1:], columns=['time_stamp'])

readings_short = readings.iloc[1800:108000+1800, :]
time_stamp_repeated = time_stamp.iloc[np.tile(np.arange(len(time_stamp)), int(len(readings_short)/len(time_stamp)))]
readings_short_equal = readings_short.iloc[:len(time_stamp_repeated), :]
readings_short_equal['time_stamp']= time_stamp_repeated['time_stamp'].values

readings_short_equal['consump_cumul'] = readings_short_equal['consump'].cumsum()
readings_short_equal['consump_pert_guass_cumul'] = readings_short_equal['consump_perturbed_guassian'].cumsum()
readings_short_equal['consump_pert_eps_cumul'] = readings_short_equal['consump_perturbed_epsilon'].cumsum()

readings_short_equal.loc[readings_short_equal['consump_perturbed_guassian'] < 0, 'consump_perturbed_guassian'] = 0
readings_short_equal.loc[readings_short_equal['consump_perturbed_epsilon'] < 0, 'consump_perturbed_epsilon'] = 0


hour_start = 1
minute_start = 35
second_start = 30
hour_end = 1
minute_end = 36
second_end = 30

start_time = hour_start*60*60+minute_start*60+second_start
end_time = hour_end*60*60+minute_end*60+second_end

consumption_d_gaussian_epsilon = pd.DataFrame({
   'original electricity consumption': readings_short_equal['consump'].tolist(),
   'perturbed electricity consumption with Gaussian': readings_short_equal['consump_perturbed_guassian'].tolist(),
   'perturbed electricity consumption with AMLODA': readings_short_equal['consump_perturbed_epsilon'].tolist()
   }, index=readings_short_equal['time_stamp'].tolist())

plt.figure(1, figsize=(8,8))
consumption_d_gaussian_epsilon.iloc[start_time:end_time,:].plot.line(figsize=(10,5))
plt.xlabel('Time')
plt.ylabel('Electricity Consumption (in watts)')

plt.savefig('electricity consumption over time for home ' + str(home_number) + ' (' + season_name + ')' + '.png')
plt.show()

consumption_cumul_d_gaussian_epsilon = pd.DataFrame({
   'original total electricity consumption': readings_short_equal['consump_cumul'].tolist(),
   'perturbed total electricity consumption with Gaussian': readings_short_equal['consump_pert_guass_cumul'].tolist(),
   'perturbed total electricity consumption with AMLODA': readings_short_equal['consump_pert_eps_cumul'].tolist()
   }, index=readings_short_equal['time_stamp'].tolist())

plt.figure(2, figsize=(8,8))
# consumption_cumul_d_gaussian_epsilon.iloc[start_time:end_time,:].plot.line(figsize=(10,5))
consumption_cumul_d_gaussian_epsilon.iloc[:,:].plot.line(figsize=(10,5))
plt.xlabel('time')
plt.ylabel('total electricity consumption')
plt.title('total electricity consumption over time for home ' + str(home_number) + ' (' + season_name + ')')
plt.savefig('total electricity consumption over time for home ' + str(home_number) + ' (' + season_name + ')' + '.png')
plt.show()


print(consumption_d_gaussian_epsilon.iloc[start_time:end_time,:])
print()
print(consumption_cumul_d_gaussian_epsilon.iloc[start_time:end_time,:])

with open(f'{home_number}_{season_name}_perturbed.csv','w') as f:
    write = csv.writer(f) 
    for j in perturbed_data_full_epsilon:
        write.writerow([j.item()]) 

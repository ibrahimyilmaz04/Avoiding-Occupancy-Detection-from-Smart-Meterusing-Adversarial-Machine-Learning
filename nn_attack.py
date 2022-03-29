import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import sys
import tensorflow_addons as tfa
import sys
from sklearn.metrics import matthews_corrcoef
np.set_printoptions(threshold=sys.maxsize)
# hyperparameters
home_number = 5
season_name = "summer"


output_size = 2
hidden_size = 150
layer_size = 2

batch_size = 10000
learning_rate = 0.001

epochs = 200

print('occupancy_dataset/home' + str(home_number) + '/' + str(home_number).zfill(2) + '_' + season_name + '.csv')

data_01 = pd.read_csv('occupancy_dataset/home' + str(home_number) + '/' + str(home_number).zfill(2) + '_' + season_name + '.csv')

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
# print(occ_dates)
plugs_01 = os.listdir('occupancy_dataset/home' + str(home_number) + '/plugs/')
plugs_readings_01 = pd.DataFrame()
target_01 = pd.DataFrame()
flage = 1

input_size = len(plugs_01)
for plug in plugs_01:

    readings_01 = pd.DataFrame()
    p = os.listdir('occupancy_dataset/home' + str(home_number) + '/plugs/' + plug)

    #loop plugs's subdirectory(ex: 01)
    for f in p:
        # print(f)
        is_common = 1
        for plg in plugs_01:
            is_common *= os.path.exists('occupancy_dataset/home' + str(home_number) + '/plugs/' + plg + '/' + f)

        if is_common:
            for i in range(len(occ_dates)):
                if occ_dates[i][:] == f[:4] + f[5:7] + f[8:10]:
                    if flage == 1:
                        target_01 = pd.concat([target_01, data_01.iloc[i, 1:]], axis=0)

                    readings_01 = pd.concat([readings_01, pd.read_csv('occupancy_dataset/home' + str(home_number) + '/plugs/' + plug + '/' + f, header=None)], sort=False)
    flage = 0
    # print(plugs_readings_01)
    plugs_readings_01 = pd.concat([plugs_readings_01, readings_01], axis=1)

count = plugs_readings_01.count()
# print(count)
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
# print(plugs_readings_01)
X,Y = drop_missisng(X,y)

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
Y.reshape(-1)
indexes0 = [i for i,j in enumerate(Y) if j[0]==0]
indexes1 = [i for i,j in enumerate(Y) if j[0]==1][:len(indexes0)]
indexes = indexes0+indexes1
new_Y = np.array([Y[i] for i in indexes])
new_X = np.array([X[i] for i in indexes])
print(np.shape(new_Y))
print(np.shape(new_X))
X_train, X_test, y_train, y_test = train_test_split(new_X, new_Y, test_size=0.1, shuffle=True, random_state=1)
# print(y_train)

# print(np.shape(X_train[0]))
# print(y_train[0])

##Neural Network
# model = tf.keras.models.Sequential()
# model.add(tf.keras.Input(shape=(30,5)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# # model.add(tf.keras.layers.Dense(64, activation='relu'))
# # model.add(tf.keras.layers.Dense(512, activation='relu'))
# model.add(tf.keras.layers.Dense(32, activation='relu'))
# model.add(tf.keras.layers.Dense(16, activation='relu'))
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# opt = keras.optimizers.Adam(lr=0.000001)
# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# model.summary()
# model.fit(X_train, y_train, epochs=100, validation_data=(X_test,y_test), batch_size=10000)
# # model.save(f"home{str(home_number)}_{season_name}.h5",save_format = 'tf')

counts = np.bincount(y_train[:, 0])
print(
    "Number of positive samples in training data: {} ({:.2f}% of total)".format(
        counts[1], 100 * float(counts[1]) / len(y_train)
    )
)

# weight_for_0 = 100000 / counts[0]
# weight_for_1 = 100000 / counts[1]
# class_weight = {0: weight_for_0, 1: weight_for_1}
x1 = layers.Input(shape=(30,6))
net = layers.Flatten()(x1)
# # net = layers.Dense(10000, activation='relu')(net)
# # net = layers.Dense(1024, activation='relu')(net)
net = layers.Dense(256, activation='relu')(net)
net = layers.Dense(64, activation='relu')(net)
net = layers.Dense(32, activation='relu')(net)
net = layers.Dense(16, activation='relu')(net)
net = layers.Dense(1, activation='sigmoid')(net)
model = Model(inputs=x1, outputs=net)
opt = keras.optimizers.Adam()
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=30000, validation_data=(X_test,y_test), batch_size=10000)
# model.save(f"home{str(home_number)}_{season_name}_new_balanced_30k.h5",save_format = 'tf')

#model = keras.Sequential()
# model.add(layers.Embedding(input_dim=150, output_dim=64))

# The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
#model.add(layers.GRU(256, return_sequences=True, input_shape = (30,5)))

# The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
#model.add(layers.SimpleRNN(128))

#model.add(layers.Dense(1, activation='sigmoid'))
#opt = keras.optimizers.Adam()
#model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#model.summary()
#model.fit(X_train, y_train, epochs=1000, validation_data=(X_test,y_test), batch_size=10000)
# model.save(f"home{str(home_number)}_{season_name}_new_balanced_10k_rnn.h5",save_format = 'tf')
kk = model.predict(X_test, batch_size=10000, verbose=1)
kk1 = model.evaluate(X_test, y_test)
print(kk1)
kk_pred = np.array([[1] if i > 0.5 else [0] for i in kk])
acc = np.mean(kk_pred == y_test)
print(f"home{str(home_number)}_{season_name}")
print("Accuracy:",acc)
mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=1)
mcc.update_state(y_test, kk_pred)
# print(matthews_corrcoef(y_test, kk_pred))
print('Matthews correlation coefficient is:',mcc.result().numpy())
y_pred =(kk>0.5)
cm = confusion_matrix(y_test, y_pred)
print(cm)
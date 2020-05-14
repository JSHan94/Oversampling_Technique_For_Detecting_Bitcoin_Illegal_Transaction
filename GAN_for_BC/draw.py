

# import library

import psutil ; print(list(psutil.virtual_memory())[0:2])

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('ggplot')

#import xgboost as xgb
import sys

#import xgboost as xgb
import pickle

import gc
gc.collect()
print(list(psutil.virtual_memory())[0:2])
base_dir = 'cache/'

# Load custom functions

import GAN_171103

# For reloading after making changes
import importlib
importlib.reload(GAN_171103) 
from GAN_171103 import *


# Load data

col_name=[  "btc_trns_avg" ,
  "btc_trns_sum" ,
  "btc_trns_min" ,
  "btc_trns_max" ,
  "btc_recv_avg" ,
  "btc_recv_sum" ,
  "btc_recv_min" ,
  "btc_recv_max" ,
  "trns_value_avg" ,
  "trns_value_sum" ,
  "trns_value_min" ,
  "trns_value_max" ,
  "recv_value_avg" ,
  "recv_value_sum" ,
  "recv_value_min" ,
  "recv_value_max" ,
  "tx_fee_trns_avg" ,
  "tx_fee_recv_avg" ,
  "sib_input_avg",
  "sib_input_out_avg",
  "sib_output_avg",
  "sib_output_in_avg",
  "sibaddr_trns_avg",
  "sibaddr_trns_out_avg",
  "sibaddr_recv_avg",
  "sibaddr_recv_in_avg",
  "txsize_trns_avg",
  "txsize_recv_avg",
  "rel_tx_trns_cnt",
  "rel_tx_recv_cnt",
  ]

exchange_data = pd.read_csv("data/exchange_59.csv", names = col_name)
silkroad_data = pd.read_csv("data/silkroad_20_21.csv", names = col_name)
#print(data.shape)




#data.head(3)

# data columns will be all other columns except class

label_cols = ['Class']
exchange_data['Class'] = 0
silkroad_data['Class'] = 1


exchange_data = exchange_data[:100000]
silkroad_data = silkroad_data[:10000]

EXCHANGE_TX_NUM = exchange_data.shape[0]
SILKROAD_TX_NUM = silkroad_data.shape[0]
print(EXCHANGE_TX_NUM)
print(SILKROAD_TX_NUM)# - (SILKROAD_TX_NUM%2))

data = pd.concat([exchange_data,silkroad_data], ignore_index=True).copy()
#data drop


#data= data.drop(['V0'],axis=1)
 
    
data_cols = list(data.columns[ data.columns != 'Class' ])
print(data_cols)
print('# of data columns: ',len(data_cols))

# 300000 normal transactions (class 0)
# 300000 fraud transactions (class 1)

data.groupby('Class')['Class'].count()



# Total nulls in dataset (sum over rows, then over columns)

data.isnull().sum().sum()


# Duplicates? Yes

normal_duplicates = sum( data.loc[ data.Class==0 ].duplicated() )
fraud_duplicates = sum( data.loc[ data.Class==1 ].duplicated() )
total_duplicates = normal_duplicates + fraud_duplicates

print( 'Normal duplicates', normal_duplicates )
print( 'Fraud duplicates', fraud_duplicates )
print( 'Total duplicates', total_duplicates )
print( 'Fraction duplicated', total_duplicates / len(data) )

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

features = ['btc_trns_avg', 'btc_trns_sum', 'btc_trns_min', 'btc_trns_max', 'btc_recv_avg', 'btc_recv_sum', 'btc_recv_min', 'btc_recv_max', 'trns_value_avg', 'trns_value_sum', 'trns_value_min', 'trns_value_max', 'recv_value_avg', 'recv_value_sum', 'recv_value_min', 'recv_value_max', 'tx_fee_trns_avg', 'tx_fee_recv_avg', 'sib_input_avg', 'sib_input_out_avg', 'sib_output_avg', 'sib_output_in_avg', 'sibaddr_trns_avg', 'sibaddr_trns_out_avg', 'sibaddr_recv_avg', 'sibaddr_recv_in_avg', 'txsize_trns_avg', 'txsize_recv_avg', 'rel_tx_trns_cnt', 'rel_tx_recv_cnt']
x = data.loc[:, features].values

y = data.loc[:,['Class']].values

x = StandardScaler().fit_transform(x)            
pca = PCA(n_components=0.99)
principalComponents = pca.fit_transform(x)
FEATURE_NUM = principalComponents.shape[1]
principalDf = pd.DataFrame(data = principalComponents,columns = ['V'+str(i) for i in range(FEATURE_NUM)])
print(principalDf)
#print(data['Class'])
#print(principalDf)
data = pd.concat([principalDf, data['Class']],axis=1).copy()
print(data.columns)
data_cols =  ['V'+str(i) for i in range(FEATURE_NUM)]
RAND_DIM = FEATURE_NUM + 1


import sklearn.cluster as cluster

train = data.loc[ data['Class']==1 ].copy()

algorithm = cluster.KMeans
args, kwds = (), {'n_clusters':2, 'random_state':0}
labels = algorithm(*args, **kwds).fit_predict(train[ data_cols ])

print( pd.DataFrame( [ [np.sum(labels==i)] for i in np.unique(labels) ], columns=['count'], index=np.unique(labels) ) )

fraud_w_classes = train.copy()
fraud_w_classes['Class'] = labels
#print(fraud_w_classes.loc[fraud_w_classes.Class == 1])

############

# reloading the libraries and setting the parameters

# Generate list of features sorted by importance in detecting fraud

# print( 'Top eight features for fraud detection: ', [ i[0] for i in sorted_x[:8] ] )


sorted_cols = ['V12', 'V4', 'V2', 'V0', 'V8', 'V1', 'V10', 'V6', 'V13', 'V3', 'V9', 'V14', 'V7', 'V5', 'V11', 'Class']

import GAN_171103
import importlib
importlib.reload(GAN_171103) # For reloading after making changes
from GAN_171103 import *
rand_dim = 15 # 32 # needs to be ~data_dim
base_n_count = 128 # 128

nb_steps = 5000 + 1 # 50000 # Add one for logging of the last interval
batch_size =30 # 64

k_d = 1  # number of critic network updates per adversarial training step
k_g = 1  # number of generator network updates per adversarial training step
critic_pre_train_steps = 100 # 100  # number of steps to pre-train the critic before starting adversarial training
log_interval = 100 # 100  # interval (in steps) at which to log loss summaries and save plots of image samples to disc
learning_rate = 5e-4 # 5e-5
data_dir = 'cache/'
generator_model_path, discriminator_model_path, loss_pickle_path = None, None, None

# show = False
show = True 

# train = create_toy_spiral_df(1000)
# train = create_toy_df(n=1000,n_dim=2,n_classes=4,seed=0)
train = fraud_w_classes.copy().reset_index(drop=True) # fraud only with labels from classification

# train = pd.get_dummies(train, columns=['Class'], prefix='Class', drop_first=True)
label_cols = [ i for i in train.columns if 'Class' in i ]

for col in train.columns:
    if col not in sorted_cols:
        sorted_cols.append(col)
sorted_cols_with_class = sorted_cols.copy()
sorted_cols.remove('Class')

data_cols = sorted_cols.copy() #[ i for i in train.columns if i not in label_cols ]

train[ data_cols ] = train[ data_cols ] / 10 # scale to random noise size, one less thing to learn
train_no_label = train[ data_cols ]



# Training the vanilla GAN and CGAN architectures

k_d = 1  # number of critic network updates per adversarial training step
learning_rate = 5e-4 # 5e-5
arguments = [rand_dim, nb_steps, batch_size, 
             k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
            data_dir, generator_model_path, discriminator_model_path, loss_pickle_path, show ]
print(train_no_label.shape)

# Let's look at some of the generated data
# First create the networks locally and load the weights

import GAN_171103
import importlib
importlib.reload(GAN_171103) # For reloading after making changes
from GAN_171103 import *

seed = 18

train = fraud_w_classes.copy().reset_index(drop=True) # fraud only with labels from classification

# train = pd.get_dummies(train, columns=['Class'], prefix='Class', drop_first=True)
label_cols = [ i for i in train.columns if 'Class' in i ]
data_cols = [ i for i in train.columns if i not in label_cols ]
train[ data_cols ] = train[ data_cols ] / 10 # scale to random noise size, one less thing to learn
train_no_label = train[ data_cols ]

data_dim = len(data_cols)
label_dim = len(label_cols)
with_class = False
if label_dim > 0: with_class = True
np.random.seed(seed)

# define network models
generator_model, discriminator_model, combined_model = define_models_CGAN(rand_dim, data_dim, label_dim, base_n_count)
generator_model.load_weights('cache/CGAN_generator_model_weights_step_5000.h5')


test_size = SILKROAD_TX_NUM # Equal to all of the fraud cases

x = get_data_batch(train, test_size, seed=17)#i+j) i,j가 무엇을 의미?
z = np.random.normal(size=(test_size, rand_dim))
print(z)
if with_class:
    labels = x[:,-label_dim:]
    g_z = generator_model.predict([z, labels])
else:
    g_z = generator_model.predict(z)



# Generate and test data with trained model
STEP = 5000
FILE_NAME = "WCGAN" + '_generator_model_weights_step_' + str(STEP) +'.h5' 
generator_model, discriminator_model, combined_model = define_models_CGAN(rand_dim, data_dim, label_dim, base_n_count, type='Wasserstein')
generator_model.load_weights( base_dir + FILE_NAME)

test_size = SILKROAD_TX_NUM
x = get_data_batch(fraud_w_classes, test_size, seed=0)
z = np.random.normal(size=(test_size, rand_dim))
labels = x[:,-label_dim:]
g_z = generator_model.predict([z, labels])



# The labels for the generate data will all be 1, as they are supposed to be fraud data


# Setup xgboost parameters

xgb_params = {
#     'max_depth': 4,
    'objective': 'binary:logistic',
    'random_state': 0,
    'eval_metric': 'auc', # auc, error
#     'tree_method': 'hist'
#     'grow_policy': 'lossguide' # depthwise, lossguide
}

# https://github.com/dmlc/xgboost/blob/master/demo/guide-python/custom_objective.py

from sklearn.metrics import recall_score, precision_score, roc_auc_score,f1_score

def recall(preds, dtrain):
    labels = dtrain.get_label()
    return 'recall',  recall_score(labels, np.round(preds))

def precision(preds, dtrain):
    labels = dtrain.get_label()
    return 'precision',  precision_score(labels, np.round(preds))

def roc_auc(preds, dtrain):
    labels = dtrain.get_label()
    return 'roc_auc',  roc_auc_score(labels, preds)

def F1(preds, dtrain):
    labels = dtrain.get_label()
    return 'f1',  f1_score(labels, preds) 

# Define model parameters

seed = 17
np.random.seed(seed)

data_dim = len(data_cols)
label_dim = len(label_cols)

base_dir = 'cache/'

base_n_count = 128



# defined training set parameters

train_fraction = 0.7
X_col = data.columns[:-1]
y_col = data.columns[-1]

folds = 5



# Function to make cross folds with different amounts of an additional dataset added

def MakeCrossFolds( g_z_df=[] ):

    np.random.seed(0)

    train_real_set, test_real_set = [], []
    train_fraud_set, test_fraud_set = [], []

    real_samples = data.loc[ data.Class==0 ].copy()
    fraud_samples = data.loc[ data.Class==1 ].copy()

#     n_temp_real = 10000 
    n_temp_real = len(real_samples)

    for seed in range(folds):
        np.random.seed(seed)

        fraud_samples = fraud_samples.sample(len(fraud_samples), replace=False).reset_index(drop=True) # shuffle

    #     n_train_fraud = int(len(fraud_samples) * train_fraction)
        n_train_fraud = 100
        train_fraud_samples = fraud_samples[:n_train_fraud].reset_index(drop=True)

    #     test_fraud_samples = fraud_samples[n_train_fraud:].reset_index(drop=True)
        n_test_fraud = SILKROAD_TX_NUM//3 # 30% left out
        test_fraud_samples = fraud_samples[-n_test_fraud:].reset_index(drop=True)

        if len(g_z_df)==0: g_z_df = fraud_samples[n_train_fraud:-n_test_fraud] # for adding real data, if no generated
        n_g_z = len(g_z_df)
        train_fraud_samples = train_fraud_samples.append(g_z_df).reset_index(drop=True)

        real_samples = real_samples.sample(len(real_samples), replace=False).reset_index(drop=True) # shuffle
        temp_real_samples = real_samples[:n_temp_real]
        n_train_real = int(len(temp_real_samples) * train_fraction)

        train_real_samples = temp_real_samples[:n_train_real].reset_index(drop=True) # with margin
        test_real_samples = temp_real_samples[n_train_real:].reset_index(drop=True) # with margin

        train_real_set.append( train_real_samples )
        test_real_set.append( test_real_samples )
        train_fraud_set.append( train_fraud_samples )
        test_fraud_set.append( test_fraud_samples )

    #print( n_train_fraud )
    #for i in [ fraud_samples, g_z_df, train_fraud_samples, test_fraud_samples ]: print( len(i) )
    #for i in [ real_samples, train_real_samples, test_real_samples ]: print( len(i) )
    # [ [ len(i) for i in j ] for j in [train_real_set, test_real_set, train_fraud_set, test_fraud_set] ]
    print("CrossFold finished!!")
    return n_train_fraud, train_real_set, test_real_set, train_fraud_set, test_fraud_set



# function to run an xgboost classifier on different cross-folds with different amounts of data added

g_z_df = pd.DataFrame( np.hstack( [g_z[:,:len(data_cols)], np.ones((len(g_z),1))] ), columns=data.columns )

n_train_fraud, train_real_set, test_real_set, train_fraud_set, test_fraud_set = MakeCrossFolds(g_z_df)
    

# Reload the testing data
t_SMOTE = pickle.load(open('cache/additional SMOTE generated fraud data test.pkl','rb'))

t_0 = pickle.load(open('cache/additional untrained generated fraud data test.pkl','rb'))
t_4800 = pickle.load(open('cache/additional generated fraud data test.pkl','rb'))
t_real = pickle.load(open('cache/additional real fraud data test.pkl','rb'))

t_DRAGAN = pickle.load(open('cache/additional DRAGAN generated fraud data test.pkl','rb'))


# Setup xgboost parameters

xgb_params = {
#     'max_depth': 4,
    'objective': 'binary:logistic',
    'random_state': 0,
    'eval_metric': 'auc', # auc, error
#     'tree_method': 'hist'
#     'grow_policy': 'lossguide' # depthwise, lossguide
}

# https://github.com/dmlc/xgboost/blob/master/demo/guide-python/custom_objective.py

from sklearn.metrics import recall_score, precision_score, roc_auc_score,f1_score

def recall(preds, dtrain):
    labels = dtrain.get_label()
    return 'recall',  recall_score(labels, np.round(preds))

def precision(preds, dtrain):
    labels = dtrain.get_label()
    return 'precision',  precision_score(labels, np.round(preds))

def roc_auc(preds, dtrain):
    labels = dtrain.get_label()
    return 'roc_auc',  roc_auc_score(labels, preds)

def F1(preds, dtrain):
    labels = dtrain.get_label()
    return 'f1',  f1_score(labels, preds) 

# Define model parameters

seed = 17
np.random.seed(seed)

data_dim = len(data_cols)
label_dim = len(label_cols)

base_dir = 'cache/'

base_n_count = 128



# defined training set parameters

train_fraction = 0.7
X_col = data.columns[:-1]
y_col = data.columns[-1]

folds = 5



# Function to make cross folds with different amounts of an additional dataset added

def MakeCrossFolds( g_z_df=[] ):

    np.random.seed(0)

    train_real_set, test_real_set = [], []
    train_fraud_set, test_fraud_set = [], []

    real_samples = data.loc[ data.Class==0 ].copy()
    fraud_samples = data.loc[ data.Class==1 ].copy()

#     n_temp_real = 10000 
    n_temp_real = len(real_samples)

    for seed in range(folds):
        np.random.seed(seed)

        fraud_samples = fraud_samples.sample(len(fraud_samples), replace=False).reset_index(drop=True) # shuffle

    #     n_train_fraud = int(len(fraud_samples) * train_fraction)
        n_train_fraud = 100
        train_fraud_samples = fraud_samples[:n_train_fraud].reset_index(drop=True)

    #     test_fraud_samples = fraud_samples[n_train_fraud:].reset_index(drop=True)
        n_test_fraud = SILKROAD_TX_NUM//3 # 30% left out
        test_fraud_samples = fraud_samples[-n_test_fraud:].reset_index(drop=True)

        if len(g_z_df)==0: g_z_df = fraud_samples[n_train_fraud:-n_test_fraud] # for adding real data, if no generated
        n_g_z = len(g_z_df)
        train_fraud_samples = train_fraud_samples.append(g_z_df).reset_index(drop=True)

        real_samples = real_samples.sample(len(real_samples), replace=False).reset_index(drop=True) # shuffle
        temp_real_samples = real_samples[:n_temp_real]
        n_train_real = int(len(temp_real_samples) * train_fraction)

        train_real_samples = temp_real_samples[:n_train_real].reset_index(drop=True) # with margin
        test_real_samples = temp_real_samples[n_train_real:].reset_index(drop=True) # with margin

        train_real_set.append( train_real_samples )
        test_real_set.append( test_real_samples )
        train_fraud_set.append( train_fraud_samples )
        test_fraud_set.append( test_fraud_samples )

    #print( n_train_fraud )
    #for i in [ fraud_samples, g_z_df, train_fraud_samples, test_fraud_samples ]: print( len(i) )
    #for i in [ real_samples, train_real_samples, test_real_samples ]: print( len(i) )
    # [ [ len(i) for i in j ] for j in [train_real_set, test_real_set, train_fraud_set, test_fraud_set] ]
    print("CrossFold finished!!")
    return n_train_fraud, train_real_set, test_real_set, train_fraud_set, test_fraud_set



# function to run an xgboost classifier on different cross-folds with different amounts of data added

# Plot the testing data

labels = ['SMOTE','DRAGAN']

metric = 'recall'
#metrics = ['recall','precision','auc','k'd,ratio,best]
plt.figure(figsize=(8,3))
#for metric in metrics:
for i, [label, test_data] in enumerate(zip(labels, [t_SMOTE, t_DRAGAN])):
    xs = [ n_train_fraud * (i[0]-1) for i in test_data.groupby('ratio') ]
    ys = test_data.groupby('ratio')[metric].mean().values
    stds = 2 * test_data.groupby('ratio')[metric].std().values

    plt.subplot(1,3,i+1)
    plt.axhline(ys[0],linestyle='--',color='red')
    plt.plot(xs,ys,c='C1',marker='o')
    plt.plot(xs,ys+stds,linestyle=':',c='C2')
    plt.plot(xs,ys-stds,linestyle=':',c='C2')
    if i==0: plt.ylabel(metric)
    plt.xlabel('# additional data')
    plt.title(label,size=12)
    plt.xlim([0,5000])
#     plt.ylim([0.15,.995])
    plt.ylim([0.25,0.35])

plt.tight_layout(rect=[0,0,1,0.9])
#plt.suptitle('Effects of additional data on fraud detection', size=16)
plt.savefig('plots/Effects of addtional data on fraud detection.png')
plt.show()

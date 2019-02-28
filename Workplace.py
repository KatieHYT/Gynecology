
# coding: utf-8

# In[1]:


import os
import sys
from datetime import datetime
import keras
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')


from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam, SGD, Adamax

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import utils as myutils
from model import build_model


# In[2]:





# In[3]:


#### parser
parser = argparse.ArgumentParser()
parser.add_argument('-d' ,'--data_dir', type=str, default='/home/katieyth/gynecology/data/', help='data directory')
parser.add_argument('-s' ,'--model_save', type=str, default='/home/katieyth/gynecology/model_save/', help='model save path')
parser.add_argument('-y' ,'--target', type=str, default='UA', help='prediction target')
# variability	UA	 deceleration management

# input parameter
parser.add_argument('-th','--acceptable_zeros_threshold', type=float, default=200, help='acceptable number of missing values in raw data')
parser.add_argument('-l' ,'--length', type=int, default=600, help='length of input')
parser.add_argument('-ks','--k_slice', type=int, default=1, help='a input will be sliced into k_slice segments when testing')
parser.add_argument('-c' ,'--n_channel', type=int, default=2, help='number of input channels')
parser.add_argument('-rn','--random_noise', type=int, default=0, help='add Gaussian noise (mean=0, std=0.01) into inputs')
parser.add_argument('-nm','--normalized', type=int, default=1, help='whether conduct channel-wise normalization')
parser.add_argument('-fb' ,'--force_binary', type=int, default=0, help='force to binary task')
parser.add_argument('-sb' ,'--split_by_patient', type=int, default=1, help='split by patient')

# data augmentation 
parser.add_argument('-aug_fliplr' ,'--aug_fliplr', type=int, default=0, help='reverse time series')
parser.add_argument('-shift' ,'--DA_Shift', type=int, default=1, help='')
parser.add_argument('-scale' ,'--DA_Scale', type=int, default=1, help='')
parser.add_argument('-randsamp' ,'--DA_RandSampling', type=int, default=1, help='')


# model parameters
parser.add_argument('-k' ,'--kernel_size', type=int, default=3, help='kernel size')
parser.add_argument('-f' ,'--filters', type=int, default=64, help='base number of filters')
parser.add_argument('-ly' ,'--layers', type=int, default=10, help='number of residual layers')
parser.add_argument('-a' ,'--activation', type=str, default='relu', help='activation function')
parser.add_argument('-i' ,'--kernel_initializer', type=str, default='RandomNormal', help='kernel initialization method')
parser.add_argument('-l2','--l2', type=float, default=0.01, help='coefficient of l2 regularization')

# hyper-parameters
parser.add_argument('-lr','--learning_rate', type=float, default=1e-4, help='learning_rate')
parser.add_argument('-reduce_lr_patience','--reduce_lr_patience', type=int, default=5, help='reduce_lr_patience')
parser.add_argument('-bs','--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('-ep','--epoch', type=int, default=1000, help='epoch')
parser.add_argument('-wb','--weight_balance', type=int, default=1, help='whether weight balancing or not')
parser.add_argument('-mntr','--monitor', type=str, default='val_acc', help='val_acc or val_loss')


parser.add_argument('-g' ,'--gpu_id', type=str, default='4', help='GPU ID')
parser.add_argument('-rs' ,'--random_state', type=int, default=13, help='random state when train_test_split')
parser.add_argument('-fn' ,'--summary_file', type=str, default=None, help='summary filename')


FLAG = parser.parse_args()

#train(FLAG)


# In[4]:


# %load utils.py
import os
import keras
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences

MEAN_B, STD_B = 138.712, 16.100
MEAN_M, STD_M =  36.346, 25.224

def Probability(a):
    def prob():
        return random.uniform(a=0, b=1) > a
    return prob

def DA_RandomNoise(x):
    noise = np.array([[random.gauss(mu=0, sigma=0.01), 
                       random.gauss(mu=0, sigma=0.01)] for _ in range(x.shape[0])], dtype=np.float32)
    return x + noise

def DA_Shift(x, smin=-5, smax=5):
    shift = np.around(random.uniform(a=smin, b=smax))
    return x + shift

def DA_Scale(x, smin=0.8, smax=1.2):
    scale = random.uniform(a=smin, b=smax)
    return np.round(x*scale)

def DA_TimeWarp(X, sigma=0.2):
    length = X.shape[0]
    channel = X.shape[1]
    knot = 4
    
    from scipy.interpolate import CubicSpline      # for warping

    xx = (np.ones((channel,1))*(np.arange(0, length, (length-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, channel))
    x_range = np.arange(length)
    cs_x = CubicSpline(xx[:,0], yy[:,0])
    tt = np.array([cs_x(x_range)]).transpose()

    tt_new = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(length-1)/tt_new[-1,0]]
    tt_new[:,0] = tt_new[:,0]*t_scale[0]

    # tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(length)
    
    X_new[:,0] = np.interp(x_range, tt_new[:,0], X[:,0])
    X_new[:,1] = np.interp(x_range, tt_new[:,0], X[:,1])
    return X_new

def RandSampleTimesteps(X, nSample=150):
    X_new = np.zeros(X.shape)
    tt = np.zeros((nSample,X.shape[1]), dtype=int)
    tt[1:-1,0] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
    tt[1:-1,1] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
    #tt[1:-1,2] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
    tt[-1,:] = X.shape[0]-1
    return tt
def DA_RandSampling(X, nSample=150):  # could for UA 
    tt = RandSampleTimesteps(X, nSample)
    X_new = np.zeros(X.shape)
    X_new[:,0] = np.interp(np.arange(X.shape[0]), tt[:,0], X[tt[:,0],0])
    X_new[:,1] = np.interp(np.arange(X.shape[0]), tt[:,1], X[tt[:,1],1])
    #X_new[:,2] = np.interp(np.arange(X.shape[0]), tt[:,2], X[tt[:,2],2])
    return X_new




def data_preprocess(x,  normalized, random_noise, aug_func=[], prob=0.5):
    do_or_not = Probability(prob)
    
    length = x.shape[0]
    # get x and then remove zeros (no info)
    x = x[(x[:,0] > 0.0) * (x[:,1] > 0.0)]
    
    # add random_noise
    if aug_func:
        for func in aug_func:
            if do_or_not():
                x = func(x)
    
    if normalized:
        x[:,0] = (x[:,0] - MEAN_B)/STD_B
        x[:,1] = (x[:,1] - MEAN_M)/STD_M

    
    if random_noise:
        x1, x2 = np.mean(x, axis=0)
        noise = np.array([[random.gauss(mu=0, sigma=0.01), 
                           random.gauss(mu=0, sigma=0.01)] for _ in range(x.shape[0])], dtype=np.float32)
        x = x + noise

    # transpose to (n_channel, arbitrary length), then padd to (n_channel, length)
    x = pad_sequences(np.transpose(x), padding='post', value=0.0, maxlen=length, dtype=np.float)

    # transpose back to original shape and store
    return np.transpose(x)


def k_slice_X(Xvalid, Yvalid, k_slice=5, length=300, class_weight = {}):
    """
    # moving across a sequence, we slice out "k_slice" segments with a constant interval
    # in order to increase validation data
    # ex:  |------------------|
    # 1    |------|
    # 2       |------|
    # 3          |------|
    # 4             |------|
    # 5                |------|
    """
    if not class_weight:
        class_weight = dict()
        for i in range(Yvalid.shape[1]):
            class_weight[i] = 1

    intvl = (Xvalid.shape[1] - length)//k_slice

    Xtest = np.empty((Xvalid.shape[0]*k_slice, length, Xvalid.shape[2]))
    Ytest = np.empty((Yvalid.shape[0]*k_slice, Yvalid.shape[1]))
    Wtest = np.empty((Yvalid.shape[0]*k_slice,))

    for k in range(k_slice):
        st = k * Xvalid.shape[0]
        for i in range(Xvalid.shape[0]):
            # print(st+i)
            Xtest[st+i,:,:] = data_preprocess(Xvalid[i,k*intvl:(k*intvl+length),:])
            Ytest[st+i,:] = Yvalid[i,:]
            Wtest[st+i] = class_weight[np.argmax(Yvalid[i,:])]
    return Xtest, Ytest, Wtest

def get_n_zeros(d):
    n_zeros = list()
    for i in range(d.shape[0]):
        n_zeros.append(sum(d[i,:] ==0))
    return np.array(n_zeros)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_keras_csv_logger(csv_logger, save_dir='', accuracy=False):
    loss = pd.read_table(csv_logger.filename, delimiter=',')
    print('min val_loss {0} at epoch {1}'.format(min(loss.val_loss), np.argmin(loss.val_loss)))
    plt.plot(loss.epoch, loss.loss, label='loss')
    plt.plot(loss.epoch, loss.val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()

    if accuracy:
        print('max val_accu {0} at epoch {1}'.format(max(loss.val_acc), np.argmax(loss.val_acc)))
        plt.plot(loss.epoch, loss.acc, label='accu')
        plt.plot(loss.epoch, loss.val_acc, label='val_accu')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.savefig(os.path.join(save_dir, 'accu.png'))
        plt.close()


# In[5]:


print("===== create directory =====")
model_id = FLAG.target + "_" + datetime.now().strftime("%y%m%d%H%M%S")
model_save = os.path.join(FLAG.model_save, FLAG.target, model_id)
summary_save = os.path.join(FLAG.model_save, FLAG.target, 'summary_'+FLAG.target+'.csv')

if not os.path.exists(model_save):
    os.makedirs(model_save)
    print(model_save)
    
# if not os.path.exists(model_save):
#     os.mkdir(model_save)
#     print('directory {0} is created.'.format(model_save))
# else:
#     print('directory {0} already exists.'.format(model_save))


# In[6]:


print("===== train =====")
os.environ['CUDA_VISIBLE_DEVICES'] = FLAG.gpu_id

d = pd.read_csv(os.path.join(FLAG.data_dir, 'data_merged.csv'))
d = d[myutils.get_n_zeros(np.array(d[[k for k in d.columns if 'b-' in k]], dtype=np.float)) <= FLAG.acceptable_zeros_threshold]

if FLAG.force_binary : 
    d[d[FLAG.target]>1] = 1

n_classes = len(set(d[FLAG.target]))

# replace 0 (no readings) with np.nan for later substitution
for k in d.columns:
    if 'b-' in k or 'm-' in k:
        print(k, end='\r')
        d.loc[d[k]==0, k] = np.nan

# train test split

if FLAG.split_by_patient:
    train_id, valid_id = train_test_split(list(set(d.ID)), test_size=0.3, random_state=FLAG.random_state)
    train_d, valid_d = d[[k in set(train_id) for k in d.ID]], d[[k in set(valid_id) for k in d.ID]]
else:
    train_d,valid_d = train_test_split(d, test_size=0.3, random_state=FLAG.random_state, stratify =d[FLAG.target])

# interpolate missing values
train_db = np.array(train_d[[k for k in train_d.columns if 'b-' in k]].interpolate(limit_direction='both', axis=1), dtype=np.float)
train_dm = np.array(train_d[[k for k in train_d.columns if 'm-' in k]].interpolate(limit_direction='both', axis=1), dtype=np.float)

valid_db = np.array(valid_d[[k for k in valid_d.columns if 'b-' in k]].interpolate(limit_direction='both', axis=1), dtype=np.float)
valid_dm = np.array(valid_d[[k for k in valid_d.columns if 'm-' in k]].interpolate(limit_direction='both', axis=1), dtype=np.float)

# combine signals from baby and mom
Xtrain = np.stack([train_db, train_dm], axis=2)
Xvalid = np.stack([valid_db, valid_dm], axis=2)

# convert labels to one-hot encodings
Ytrain = keras.utils.to_categorical(np.array(train_d[FLAG.target]), num_classes=n_classes)
Yvalid = keras.utils.to_categorical(np.array(valid_d[FLAG.target]), num_classes=n_classes)

# weight balancing or not
if FLAG.weight_balance:

    y_integers = np.argmax(Ytrain, axis=1)
    d_class_weight = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    class_weight = dict(enumerate(d_class_weight))
    print('class weight: {0}'.format(class_weight))
else:
    class_weight = dict()
    for i in range(n_classes):
        class_weight[i] = 1

# k fold of validation set
Xtest, Ytest, Wtest = myutils.k_slice_X(Xvalid, Yvalid, length=FLAG.length, k_slice=FLAG.k_slice, class_weight = class_weight)


# In[7]:


train_d.groupby(FLAG.target)[FLAG.target].count()


# In[8]:


valid_d.groupby(FLAG.target)[FLAG.target].count()


# In[9]:


def my_generator(Xtrain, Ytrain, length=300, n_channel=2, n_classes=2, batch_size=16, prob=0.5, aug_func=[], random_noise = False, normalized = True):
    n_sample = Xtrain.shape[0]
    n_length = Xtrain.shape[1]
    ind = list(range(n_sample))
    x = np.empty((batch_size, length, n_channel), dtype=np.float)
    y = np.empty((batch_size, n_classes), dtype=int)

    while True:
        np.random.shuffle(ind)
        for i in range(n_sample//batch_size):
            if length==600:
                st = 0
            else:
                st = random.choice(np.arange(0, Xtrain.shape[1] - length))
            i_batch = ind[i*batch_size:(i+1)*batch_size]
            for j, k in enumerate(i_batch):
                x[j,:] = data_preprocess(Xtrain[k,st:(st+length),:], aug_func=aug_func, prob=prob, random_noise=random_noise, normalized=normalized)
                y[j,:] = Ytrain[k,:]
            yield x, y


# In[10]:


# declare model 
model = build_model(length=FLAG.length, n_channel=FLAG.n_channel, n_classes=n_classes, filters=FLAG.filters, kernel_size=FLAG.kernel_size, layers=FLAG.layers,
                activation=FLAG.activation, kernel_initializer=FLAG.kernel_initializer, l_2=FLAG.l2)
model.summary()


lr_rate = FLAG.learning_rate
adam = Adamax(lr_rate, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay = 0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

csv_logger = keras.callbacks.CSVLogger(os.path.join(model_save, 'training.log'))
checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(model_save, 'model.h5'), 
                                            monitor=FLAG.monitor, 
                                            verbose=1, 
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto',
                                            period=1)
earlystop = EarlyStopping(monitor = FLAG.monitor, patience=20, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor=FLAG.monitor, factor = 0.5, patience = FLAG.reduce_lr_patience, min_lr = 0, cooldown = 5, verbose = True)


# In[11]:


if FLAG.aug_fliplr:
    Xtrain_copy = Xtrain.copy()
    for i in range(len(Xtrain)):
        Xtrain_copy[i] = np.fliplr([Xtrain[i]])[0]
#         plt.plot(Xtrain[i])
#         plt.show()
#         plt.plot(Xtrain_copy[i])
#         plt.show()
        print(i,'/',len(Xtrain), end= '\r')
    Xtrain = np.vstack((Xtrain, Xtrain_copy))
    Ytrain = np.vstack((Ytrain, Ytrain))


# In[12]:


augtool = (DA_Shift,DA_Scale,DA_RandSampling)
choose_augtool = (FLAG.DA_Shift,FLAG.DA_Scale,FLAG.DA_RandSampling)
augset = [x for x, y in zip(augtool, choose_augtool) if y == 1]
augset


# In[13]:


# fit
model.fit_generator(generator=my_generator(Xtrain, Ytrain, 
                                        length=FLAG.length, 
                                        n_channel=FLAG.n_channel, 
                                        n_classes=n_classes,
                                        random_noise=FLAG.random_noise,
                                        normalized=FLAG.normalized,
                                        batch_size=FLAG.batch_size,
                                        aug_func=augset,
                                        prob=0.25),
                    class_weight=class_weight,
                    validation_data=(Xtest, Ytest, Wtest),
                    steps_per_epoch=50, 
                    epochs=FLAG.epoch,
                    verbose=1,
                    callbacks=[csv_logger,
                            #reduce_lr, 
                            checkpoint,
                            #earlystop
                              ])


# In[ ]:





# In[ ]:


# plot csv logger
myutils.plot_keras_csv_logger(csv_logger, save_dir=model_save, accuracy=True)


# In[ ]:


# evaluate validation set
trained_model = load_model(os.path.join(model_save,'model.h5'))
Pred = trained_model.predict(Xtest)


# In[ ]:


# evaluate by every segment
ypred_aug = np.argmax(Pred , axis=1)
ytest_aug = np.argmax(Ytest, axis=1)

cfm = confusion_matrix(y_pred=ypred_aug, y_true=ytest_aug)

plt.figure()
myutils.plot_confusion_matrix(cfm, classes=np.arange(n_classes), title='Confusion matrix, without normalization')
plt.savefig(os.path.join(model_save, 'segment_confusion_matrix.png'))
plt.close()


# In[ ]:


import collections

# aggregate by voting
#ypred = np.ceil(np.mean(ypred_aug.reshape(FLAG.k_slice,-1), axis=0))
ypred = np.round(np.mean(ypred_aug.reshape(FLAG.k_slice,-1), axis=0), 0)
#ypred = (np.mean(ypred_aug.reshape(FLAG.k_slice,-1), axis=0) > 0.5) + 0 # voting
ytest = np.argmax(Yvalid, axis=1)


# calculate aggregated results
cfm = confusion_matrix(y_pred=ypred, y_true=ytest)
recall = np.diag(cfm) / np.sum(cfm, axis=1)
precision = np.diag(cfm) / np.sum(cfm, axis=0)
vote_val_accu = accuracy_score(y_pred=ypred, y_true=ytest)


for i in range(n_classes):
    print('recall-{0} : {1}'.format(i, recall[i]))
    #sav['precision-{0}'.format(i)] = precision[i]
    

plt.figure()
myutils.plot_confusion_matrix(cfm, classes=np.arange(n_classes), title='Confusion matrix, without normalization')
plt.savefig(os.path.join(model_save, 'voting_confusion_matrix.png'))
plt.close()


# calculate average accuracy from segments
# and voting accuracy
tmp = ypred_aug.reshape(FLAG.k_slice,-1)
savg_val_accu = 0.0
for i in range(tmp.shape[0]):
    accu = accuracy_score(y_pred=tmp[i,:], y_true=ytest)
    print('{0}-segment accuracy={1}'.format(i, accu))
    savg_val_accu += accu
savg_val_accu /= tmp.shape[0]
print('avg accu={0}'.format(savg_val_accu))
print('vote accu={0}'.format(vote_val_accu))

class_ratio = collections.Counter(ytest)
for i in range(len(class_ratio)):
    print('Ytest ratio class-%s: %s' % (i, class_ratio[i]/len(ytest)))


# In[ ]:


# read traing.log
loss = pd.read_table(csv_logger.filename, delimiter=',')
best_val_loss = np.min(loss.val_loss)
best_epoch = np.argmin(loss.val_loss)


# In[ ]:


# save into dictionary
sav = vars(FLAG)
sav['epoch'] = best_epoch
sav['val_loss'] = best_val_loss
sav['vote_val_accu'] = vote_val_accu
sav['savg_val_accu'] = savg_val_accu
sav['model_id'] = model_id

for i in range(n_classes):
    sav['recall-{0}'.format(i)] = recall[i]
    sav['precision-{0}'.format(i)] = precision[i]

# append into summary files
dnew = pd.DataFrame(sav, index=[0])
if os.path.exists(summary_save):
    dori = pd.read_csv(summary_save)
    dori = pd.concat([dori, dnew])
    dori.to_csv(summary_save, index=False)
else:
    dnew.to_csv(summary_save, index=False)

print(summary_save)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





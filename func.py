import os
import random
import itertools
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences

MEAN_B, STD_B = 138.712, 16.100
MEAN_M, STD_M =  36.346, 25.224


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Directory created: %s' %(path))
        
def DA_Shift(x, smin=-5, smax=5):
    shift = np.around(random.uniform(a=smin, b=smax))
    return x + shift

def DA_Scale(x, smin=0.8, smax=1.2):
    scale = random.uniform(a=smin, b=smax)
    return np.round(x*scale)
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
def data_normalize(X,  normalized = True):

    if normalized:
        x = X.copy()
        x[:,0] = (x[:,0] - MEAN_B)/STD_B
        x[:,1] = (x[:,1] - MEAN_M)/STD_M
    return x


def data_preprocess_test(Xvalid, Yvalid, length=600, class_weight = {}):
    
    if not class_weight:
        class_weight = dict()
        for i in range(Yvalid.shape[1]):
            class_weight[i] = 1

    Xtest = np.empty((Xvalid.shape[0], length, Xvalid.shape[2]))
    Ytest = np.empty((Yvalid.shape[0], Yvalid.shape[1]))
    Wtest = np.empty((Yvalid.shape[0],))

    for i in range(Xvalid.shape[0]):
        # print(st+i)
        Xtest[i,:,:] = data_normalize(Xvalid[i,0:600,:])
        Ytest[i,:] = Yvalid[i,:]
        Wtest[i] = class_weight[np.argmax(Yvalid[i,:])]
    return Xtest, Ytest, Wtest


def Probability(a):
    def prob():
        return random.uniform(a=0, b=1) > a
    return prob

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
            
            
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save_dir=''):
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
    if not save_dir =='':
        plt.savefig(os.path.join(save_dir, 'cm.png'))

    
def plot_keras_csv_logger(csv_logger, save_dir='', accuracy=False):
    if type(csv_logger) is str:
        loss = pd.read_table(csv_logger, delimiter=',')
    else:
        loss = pd.read_table(csv_logger.filename, delimiter=',')
    print('min val_loss {0} at epoch {1}'.format(min(loss.val_loss), np.argmin(loss.val_loss)))
    plt.plot(loss.epoch, loss.loss, label='loss')
    plt.plot(loss.epoch, loss.val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.show()
    plt.close()

    if accuracy:
        print('max val_accu {0} at epoch {1}'.format(max(loss.val_acc), np.argmax(loss.val_acc)))
        plt.plot(loss.epoch, loss.acc, label='accu')
        plt.plot(loss.epoch, loss.val_acc, label='val_accu')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.savefig(os.path.join(save_dir, 'accu.png'))
        plt.show()
        plt.close()
     
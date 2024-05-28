from csv import writer
import pandas as pd
import numpy as np
import math
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Add
from keras.layers import Flatten, Activation, Lambda
from keras import backend as K
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.models import load_model
from Lite_SeqCNN_encoder import *
from Evaluate import *
import matplotlib.pyplot as plt
np.random.seed(7)

def test_segment(filename, low, up):
    myFile = open(filename, 'w', newline = '')
    with myFile:
        csv_writer = writer(myFile)
        for j, row in enumerate(seqData):
            segment = [ ]
            if(len(row) > low and len(row) < up):
                segment.append(row)
                for item in label[j]:
                    segment.append(item)
                csv_writer.writerow(segment)
    myFile.close()

dataframe = pd.read_csv("/content/gdrive/MyDrive/CAFA3/bp/test_data_bp1.csv", header=None)
dataset = dataframe.values
seqData = dataset[:,0]
label = dataset[:,1:len(dataset[0])]
print('Original Dataset Size : %s' %len(dataset))
test_segment('testData200.csv', 0, 201)
test_segment('testData500.csv', 200, 501)
test_segment('testData1000.csv', 500, 1001)
test_segment('testData16000.csv', 1000, 16000)


def cls_predict(pred, normalize=True, sample_weight=None):
    s_mean = np.mean(pred, axis=0)
    m = max(s_mean)
    s_mean = (s_mean/m)
    return(list(s_mean))

def final_model(filename, segSize, nonOL,filter_size):
    max_seq_len = segSize - chunkSize + 1
    overlap = 50
 
    model_path = '/content/gdrive/MyDrive/CAFA_C/bp/D_CNN/ablation/'+str(64)+'_model_'+str(1280)+'_'+str(nonOL)+'_'+str(filter_size)+'_'+ str(segSize) +'.h5'
    print(model_path)
    model = load_model(model_path, compile = False)
    print(model.summary())

    print('Extracting features based on LSTM model...... ')
    dataframe2 = pd.read_csv(filename, header=None)
    dataset2 = dataframe2.values
    X_test = dataset2[:,0]
    Y_test = dataset2[:,1:len(dataset2[0])]

    c_p = []
    for tag, row in enumerate(X_test):
        pos = math.ceil(len(row) / overlap)
        if(pos < math.ceil(segSize/ overlap)):
            pos = math.ceil(segSize/ overlap)
        segment = [ ]
        for itr in range(pos - math.ceil(segSize/overlap) + 1):
            init = itr * overlap
            segment.append(row[init : init + segSize])
        seg_nGram = nGram(segment, chunkSize, dict_Prop)
        test_seg = sequence.pad_sequences(seg_nGram, maxlen=max_seq_len)
        preds = model.predict(test_seg)
        c_p.append(cls_predict(preds))
    c_p = np.array(c_p)

    del model
    return c_p, Y_test

def create_nn_model(dim):
    n_model = Sequential()
    n_model.add(Dense(dim, input_dim = dim, kernel_initializer='normal', activation='relu'))
    n_model.add(Dense(dim, kernel_initializer='normal', activation='sigmoid'))
    n_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return n_model

# Creates a HDF5 file 'my_model.h5'

X_train_new1, Y_train_new = final_model("/content/gdrive/MyDrive/CAFA3/bp/train_data_bp1.csv", 200, 150,6)
model11 = create_nn_model(Y_train_new[0].shape[0])
print(model11.summary())
early_stopping_monitor = EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1)
model11.fit(X_train_new1, Y_train_new.astype(None),
           callbacks = [early_stopping_monitor],
           validation_split = 0.1,
           epochs = 1000,
           batch_size = 150,
           verbose = True)

X_train_new2, _ = final_model("/content/gdrive/MyDrive/CAFA3/bp/train_data_bp1.csv", 300, 250,6)
model12 = create_nn_model(Y_train_new[0].shape[0])
print(model12.summary())
early_stopping_monitor = EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1)
model12.fit(X_train_new2, Y_train_new.astype(None),
           callbacks = [early_stopping_monitor],
           validation_split = 0.1,
           epochs = 1000,
           batch_size = 150,
           verbose = True)

X_train_new3, _ = final_model("/content/gdrive/MyDrive/CAFA3/bp/train_data_bp1.csv", 400, 350,6)
model13 = create_nn_model(Y_train_new[0].shape[0])
print(model13.summary())
early_stopping_monitor = EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1)
model13.fit(X_train_new3, Y_train_new.astype(None),
           callbacks = [early_stopping_monitor],
           validation_split = 0.1,
           epochs = 1000,
           batch_size = 150,
           verbose = True)

def final_model(filename, segSize, nonOL,filter_size):
    max_seq_len = segSize - chunkSize + 1
    overlap = 50

    model_path = '/content/gdrive/MyDrive/CAFA_C/bp/D_CNN/ablation/'+str(64)+'_model_'+str(1280)+'_'+str(nonOL)+'_'+str(filter_size)+'_'+ str(segSize) +'.h5'  
    model = load_model(model_path)
    print(model.summary())

    print('Extracting features based on LSTM model...... ')
    dataframe2 = pd.read_csv(filename, header=None)
    dataset2 = dataframe2.values
    X_test = dataset2[:,0]
    Y_test = dataset2[:,1:len(dataset2[0])]

    c_p = []
    for tag, row in enumerate(X_test):
        pos = math.ceil(len(row) / overlap)
        if(pos < math.ceil(segSize/ overlap)):
            pos = math.ceil(segSize/ overlap)
        segment = [ ]
        for itr in range(pos - math.ceil(segSize/overlap) + 1):
            init = itr * overlap
            segment.append(row[init : init + segSize])
        seg_nGram = nGram(segment, chunkSize, dict_Prop)
        test_seg = sequence.pad_sequences(seg_nGram, maxlen=max_seq_len)
        preds = model.predict(test_seg)
        c_p.append(cls_predict(preds))
    c_p = np.array(c_p)

    del model
    return c_p, Y_test

from matplotlib import pyplot as plt

# Testing
def test_fun(file):
    X_test_new1, Y_test_new = final_model(file, 200, 150, 6)
    X_test_new2, _ = final_model(file, 300,250,6)
    X_test_new3, _ = final_model(file, 400,350,6)

    print(X_test_new1.shape, Y_test_new.shape)
    print(X_test_new2.shape, Y_test_new.shape)
    print(X_test_new3.shape, Y_test_new.shape)
    Y_test_new = np.array(Y_test_new).astype(None)

    fmax, tmax = 0.0, 0.0
    precisions, recalls = [], []
    for t in range(1, 101, 1):
        test_preds1 = model11.predict(X_test_new1)
        test_preds2 = model12.predict(X_test_new2)
        test_preds3 = model13.predict(X_test_new3)
        test_preds = (test_preds1 + test_preds2 + test_preds3) / 3   #Average of all the predictions

        threshold = t / 100.0
        print("THRESHOLD IS =====> ", threshold)
        test_preds[test_preds>=threshold] = int(1)
        test_preds[test_preds<threshold] = int(0)

        rec = recall(Y_test_new, test_preds)
        pre = precision(Y_test_new, test_preds)
        if math.isnan(pre):
            pre = 0.0
        recalls.append(rec)
        precisions.append(pre)

        f1 = f_score(Y_test_new, test_preds)*100
        f = 2 * pre * rec / (pre + rec)
        print('Recall: {0}'.format(rec*100), '     Precision: {0}'.format(pre*100),
              '     F1-score1: {0}'.format(f*100), '      F1-score2: {0}'.format(f1))

        if fmax < f:
            fmax = f
            tmax = threshold
    
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    print(f'AUPR: {aupr:0.3f}')

    plt.figure()
    plt.plot(recalls, precisions, color='darkorange', lw=2, label=f'AUPR curve (area = {aupr:0.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Area Under the Precision-Recall curve')
    plt.legend(loc="upper right")
    plt.savefig(f'aupr.pdf')

    return tmax

th_set = test_fun("/content/gdrive/MyDrive/CAFA3/bp/test_data_bp1.csv")
print("Best Threshold: ", th_set)

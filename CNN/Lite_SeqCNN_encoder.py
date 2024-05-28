import numpy as np
import datetime
from csv import writer
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math
import math
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from Segmentation import segmentation
from Evaluate import *

np.random.seed(7)

# Preparing For Training
segmentSize = 200
nonOL = segmentSize - 50
SEG = str(segmentSize)

X, Y = segmentation(X, Y, segmentSize, nonOL)
nb_of_cls = len(Y[0])

#Split the dataset
x_tr, x_val, y_tr, y_val = train_test_split(X, Y, test_size = 0.1, random_state = 42)
del X, Y

y_train = np.array(y_tr, dtype=float)
y_validate = np.array(y_val, dtype=float)
print(len(x_tr), len(x_val))
print(y_train.shape, y_validate.shape)

del y_tr, y_val


def dictionary(chunk_size):
    dataframe = pd.read_csv('/content/gdrive/MyDrive/CAFA3/bp/train_data_bp1.csv', header=None)
    dataset = dataframe.values
    del dataframe

    seq_dataset = dataset[:,0]
    print('Creating Dictionary:')
    dict = {}
    j = 0
    for row in seq_dataset:
        for i in range(len(row) - chunk_size + 1):
            key = row[i:i + chunk_size]
            if key not in dict:
                dict[key] = j
                j = j + 1
    del dataset, seq_dataset
    return(dict)

def nGram(dataset, chunk_size, dictI):
    dict1 = list()
    for j, row in enumerate(dataset):
        string = row
        dict2 = list()
        for i in range(len(string) - chunk_size + 1):
            try:
                dict2.append(dictI[string[i:i + chunk_size]])
            except:
                None
        dict1.append(dict2)   
    return(dict1)

# CREATING DICTIONARY
chunkSize = 4
dict_Prop = dictionary(chunkSize)
max_seq_len = segmentSize - chunkSize + 1

def DC_CNN_Block(nb_filter, filter_length, dilation, l2_layer_reg):
    def f(input_):
        residual = input_
        layer_out = layers.Conv1D(filters=nb_filter, kernel_size=filter_length, dilation_rate=dilation, 
                                  activation='linear', padding='same', use_bias=True) (input_)
        layer_out = layers.BatchNormalization(epsilon=1.1e-5)(layer_out)
        layer_out = layers.LeakyReLU(alpha = 0.2)(layer_out)
        return layer_out
    return f

embed_dim = 32
ff_dim = 1280

def DC_CNN_Model(top_words, seq_len, o_dim):

    f_num = 256
    f_size = [6,6,6,6,6]

    _input = layers.Input(shape=(seq_len,))
    emd = layers.Embedding(top_words, embed_dim, input_length = seq_len)(_input)
    drop1 = layers.Dropout(0.3)(emd)

    l1 = DC_CNN_Block(f_num, f_size[0], 1, 0.001)(drop1) 
    l2 = DC_CNN_Block(f_num, f_size[1], 3, 0.001)(drop1)
    l3 = DC_CNN_Block(f_num, f_size[2], 5, 0.001)(drop1)
    l4 = DC_CNN_Block(f_num, f_size[3], 7, 0.001)(drop1)
    l5 = DC_CNN_Block(f_num, f_size[4], 9, 0.001)(drop1)

    x = layers.Concatenate()([l1, l2, l3, l4, l5])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    _output = layers.Dense(o_dim, kernel_initializer='normal', activation='sigmoid', name='CLASSIFIER')(x)

    model = keras.Model(inputs=_input, outputs=_output)
    model.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005),
                  metrics = tf.keras.metrics.BinaryAccuracy(threshold=0.5))
    return model

#CREATING N-GRAM
x_train = nGram(x_tr, chunkSize, dict_Prop)
x_validate = nGram(x_val, chunkSize, dict_Prop)

# truncate and pad input sequences
x_train = sequence.pad_sequences(x_train, maxlen=max_seq_len)
x_validate = sequence.pad_sequences(x_validate, maxlen=max_seq_len)

# Create & Compile the model
model = DC_CNN_Model(len(dict_Prop), max_seq_len, nb_of_cls)
early_stopping_monitor1 = EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1)
history = model.fit(x_train, y_train,
          validation_data = (x_validate, y_validate),
          epochs = 500,
          batch_size = 150,
          callbacks=[early_stopping_monitor1],
          verbose=1)

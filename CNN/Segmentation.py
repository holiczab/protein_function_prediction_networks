from csv import writer
import pandas as pd
import math

def segment(dataset, label, seg_size, overlap):
    print("Non-overlapping Region: %s" %overlap)
    print("Segment Size: %s" %seg_size)
  
    seq_data, label_data = [], []
    for j, row in enumerate(dataset):
        if(len(row) < 2001):
            pos = math.ceil(len(row)/overlap)
            if(pos < math.ceil(seg_size/overlap)):
                pos = math.ceil(seg_size/overlap)
            for itr in range(pos - math.ceil(seg_size/overlap) + 1):
                init = itr * overlap
                if(len(row[init : init + seg_size]) > 50):
                    seq_data.append(row[init : init + seg_size])
                    label_data.append(label[j])
    return seq_data, label_data

	
def segmentation(segSize, overLap):
  dataframe = pd.read_csv('/content/gdrive/MyDrive/CAFA3/bp/train_data_bp1.csv', header=None)
  dataset = dataframe.values
  print('Original Dataset Size : %s' %len(dataset))
  X = dataset[:,0]
  Y = dataset[:,1:len(dataset[0])]
  segment(X, Y, segSize, overLap)

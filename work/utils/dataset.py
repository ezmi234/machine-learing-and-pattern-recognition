import numpy as np
from sklearn.datasets import load_iris
from utils.helpers import vcol

def load_iris_dataset():
    data = load_iris()
    D = data['data'].T 
    L = data['target']
    return D, L

def load_project_dataset():
    path = '../labs/lab2/data/trainData.txt'
    DList = []
    labelsList = []

    with open(path) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = vcol(np.array([float(i) for i in attrs]))
                label = int(line.split(',')[-1].strip())
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return np.hstack(DList), np.array(labelsList, dtype=np.int32)

def split_db_2to1(D, L, seed=42):
    
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    
    return (DTR, LTR), (DVAL, LVAL)
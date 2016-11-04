import pandas as pd
from collections import Counter
from scipy.sparse import csr_matrix
import numpy as np
import scipy as sp
from numpy.linalg import norm
from random import randint
import math

dfurl = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/badges/badges.data', header=None)

def createDf(c):
    #reading file
    df = dfurl
    df.columns = ['row']
    dfOrig = df
    df = pd.DataFrame(df.row.str.split(' ',1).tolist(),columns=['class','row'])
    df['original'] = dfOrig['row']
    dfrow = pd.DataFrame(df.row.str.lower())
    df['row'] = dfrow['row']
    listoflists=[]
    for x in df.itertuples():
        alist=[]
        for i in range(0, len(x[2])-c+1):
            alist.append(x[2][i:i+c])
        listoflists.append(alist)
    df['split'] = listoflists
    #df
    return df
#createDf(2)

def createMat(c):
    df = createDf(c)
    #print(df)
    nrows = len(df)
    idx = {}
    tid = 0
    nnz = 0
    for x in df.itertuples():
        nnz += len(set(x[4]))
        for w in x[4]:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)

    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for x in df.itertuples():
        cnt = Counter(x[4])
        keys = list(ke for ke,_ in cnt.most_common())
        l = len(keys)
        for j,ke in enumerate(keys):
            ind[j+n] = idx[ke]
            val[j+n] = cnt[ke]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
        
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    #print(ind)
    #print(val)
    #print(ptr)
    return mat, df
#createMat(3)

def predictClass(pClasses):
    predic = 0
    for cls in pClasses:
        if cls == "+":
            predic = predic + 1
        else:
            predic = predic - 1
    if predic > 0:
        return "+"
    elif predic < 0:
        return "-"
    else:
        num = np.random.uniform(0,1)
        if num > 0.5:
            return "+"
        else:
            return "-"

def createFolds(k,n):
    folds=[]
    k=math.ceil(n/k)
    i=0
    folds.append(i)
    
    while(i<n):
        i=i+k
        if (i > n):
            i=n
        folds.append(i)
    return folds
        
def knnMeanAccuracy(c,k,f): #c-mers, k-nearest, f-folds
    mat, df = createMat(c)
    n = len(df)
    folds = createFolds(f,n)
    #print(folds)
    cosims=[]
    cosims = [[1.0]*n for _ in range(n)] #cosine similarities between same elements is 1
    
    for i in range(0,n):
        r1 = mat.getrow(i).toarray().reshape(-1)
        for j in range(i+1,n):
            r2 = mat.getrow(j).toarray().reshape(-1)
            cosims[i][j] = r1.dot(r2.T) / (norm(r1) * norm(r2))
            cosims[j][i] = cosims[i][j]

    j=1
    accuracies = 0.0
    while(j < len(folds)):
        knearest = {}
        accuracy = 0.0
        for a in range(folds[j-1],folds[j]):
            #print("Nearest Neighbours of " + str(df['row'][a]) + " are: ")
            for b in range(0,n):
                if b<folds[j-1] or b>=folds[j]:
                    knearest[b]=cosims[a][b]
            i=0
            pClasses=[]
            for key in sorted(knearest, key = knearest.__getitem__,reverse=True):
                if i < k:
                    #print(str(df['row'][key]) + "," + str(df['class'][key]))
                    pClasses.append(str(df['class'][key]))
                    i=i+1
            #print ("Predicted class: " + predictClass(pClasses))
            #print ("Actual class: " + str(df['class'][a]))
            pClass = predictClass(pClasses)
            if pClass == str(df['class'][a]):
                accuracy = accuracy + 1
        accuracy = accuracy/(folds[j]-folds[j-1])
        #print("Accuracy: " + str(accuracy))
        accuracies = accuracies + accuracy
        j=j+1
    #print(accuracies)
    accuracies = accuracies/(len(folds)-1)
    return accuracies
            
for c in range (1,4):
    for k in range(1,6):
        print("c,k,Mean Accuracy: " + str(c) + "," + str(k) + "," + str(knnMeanAccuracy(c,k,10)))
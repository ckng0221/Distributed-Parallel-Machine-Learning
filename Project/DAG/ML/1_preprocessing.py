#!/bin/python3
# file name: 1_preprocessing.py
import os
import datetime
import time

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def run(where='server'):
    start = time.perf_counter()

    if where == 'server':
        maindir = '/home/ec2-user/Shared' 
    elif where == 'local':
        maindir =  os.path.abspath(os.path.dirname(__file__))
    print(maindir)

    # filepath
    trainpath = os.path.join(maindir, 'data', 'fashion-mnist_train.csv')
    testpath = os.path.join(maindir, 'data', 'fashion-mnist_test.csv')

    if os.path.exists(trainpath) and os.path.exists(testpath):
        #import csv data
        traindf= pd.read_csv(trainpath)
        testdf = pd.read_csv(testpath)

        #split x and y for both train and test set
        X_train = traindf.drop(['label'], axis = 1)
        Y_train = traindf['label']

        X_test = testdf.drop(['label'], axis = 1)
        Y_test = testdf['label']

        #normalise X from 0:255 to 0:1
        X_train = X_train / 255
        X_test = X_test / 255

        #standardisation for X
        X_ss = StandardScaler()
        X_train = X_ss.fit_transform(X_train)
        X_test = X_ss.transform(X_test)

        # dimensionality reduction with pca
        pca = PCA(n_components=0.9, copy=True, whiten=False)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        #change from nparray to pd dataframe
        X_train = pd.DataFrame(X_train) 
        X_test = pd.DataFrame(X_test)

        #output csv files for X_train and Y_train for use in modelling.
        xtrainpath = os.path.join(maindir, 'data', 'xtrain.csv')
        X_train.to_csv(xtrainpath, index=False, header = True)

        ytrainpath = os.path.join(maindir, 'data', 'ytrain.csv')
        Y_train.to_csv(ytrainpath, index=False, header = True)

        xtestpath = os.path.join(maindir, 'data', 'xtest.csv')
        X_test.to_csv(xtestpath, index=False, header = True)

        ytestpath = os.path.join(maindir, 'data', 'ytest.csv')
        Y_test.to_csv(ytestpath, index=False, header = True)

        now = str(datetime.datetime.now())
        print(f"{now}: Done preprocessing.")
    else:
        print("Input dataset doesn't exits")


    end = time.perf_counter()
    timeUsed = f"--- {end - start} seconds --- "

    print(timeUsed)

if __name__ == '__main__':
    # run('local')
    run('server')

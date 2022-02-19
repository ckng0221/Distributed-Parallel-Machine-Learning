#!/bin/python3
# file name: 2c_model_svc.py
import datetime
import os
import pickle
import time

import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score

def run(where='server'):
    start_time = time.time()

    #set path
    if where == 'server':
        maindir = '/home/ec2-user/Shared' 
    elif where == 'local':
        maindir =  os.path.abspath(os.path.dirname(__file__))


    #set path
    xtrainpath = os.path.join(maindir, 'data', 'xtrain.csv')
    ytrainpath = os.path.join(maindir, 'data', 'ytrain.csv')

    if os.path.exists(xtrainpath) and os.path.exists(ytrainpath):
        #import csv data
        xtraindf = pd.read_csv(xtrainpath)
        ytraindf = pd.read_csv(ytrainpath)

        #do modelling
        model_svc = svm.SVC(gamma='scale', kernel='rbf', C=8)
        model_svc.fit(xtraindf, ytraindf.values.ravel())

        #check accuracy on train set (can remove)
        ytrain_prd = model_svc.predict(xtraindf)
        acc_train_svc = accuracy_score(ytraindf, ytrain_prd)

        # ensure having model folder
        modelDir = os.path.join(maindir, "model")
        if not os.path.exists(modelDir): 
            os.mkdir(modelDir)

        #output model_svc to model fir with pickle
        svcpath = os.path.join(modelDir, "model_svc.pkl")
        svcfile = open(svcpath, 'wb')
        pickle.dump(model_svc, svcfile)
        svcfile.close()

        print("SVC: accuracy on train set:{:.4f}".format(acc_train_svc))
        print("--- %s seconds --- \n" % (time.time() - start_time))
        now = str(datetime.datetime.now())
        print(f"{now}: Done SVC")


    else:
        print("Input dataset doesn't exits")


if __name__ == '__main__':
    # run('local')
    run('server')

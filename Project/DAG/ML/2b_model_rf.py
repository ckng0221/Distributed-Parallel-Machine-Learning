#!/bin/python3
# file name: 2b_model_rf.py
import datetime
import os
import pickle
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
        #model_rf = RandomForestClassifier(max_features='auto', oob_score=True, random_state=1, n_jobs=-1)
        model_rf  = RandomForestClassifier(criterion='gini', n_estimators=700, min_samples_split=10, min_samples_leaf=1,
        max_features='auto', oob_score=True, random_state=1, n_jobs=-1)
        model_rf.fit(xtraindf, ytraindf.values.ravel())

        #check accuracy on train set (can remove)
        ytrain_prd = model_rf.predict(xtraindf)
        acc_train_rf = accuracy_score(ytraindf, ytrain_prd)

        # ensure having model folder
        modelDir = os.path.join(maindir, "model")
        if not os.path.exists(modelDir): 
            os.mkdir(modelDir)

        #output model_svc to model fir with pickle
        rfpath = os.path.join(modelDir, "model_rf.pkl")
        rffile = open(rfpath, 'wb')
        pickle.dump(model_rf, rffile)
        rffile.close()

        print("RF: accuracy on train set:{:.4f}".format(acc_train_rf))
        print("--- %s seconds --- \n" % (time.time() - start_time))
        now = str(datetime.datetime.now())
        print(f"{now}: Done Random Forest")


    else:
        print("Input dataset doesn't exist")


if __name__ == '__main__':
    # run('local')
    run('server')

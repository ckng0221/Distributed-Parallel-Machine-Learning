#!/bin/python3
# file name: 3_res_eval.py
import datetime
import os
import pickle
import time

import pandas as pd
from sklearn.metrics import accuracy_score

def run(where='server'):
    start = time.perf_counter()

    #set path
    if where == 'server':
        maindir = '/home/ec2-user/Shared' 
    elif where == 'local':
        maindir =  os.path.abspath(os.path.dirname(__file__))

    # train data path
    xtestpath = os.path.join(maindir, 'data', 'xtest.csv')
    ytestpath = os.path.join(maindir, 'data', 'ytest.csv')

    #load model
    knnpath = os.path.join(maindir, "model", "model_knn.pkl")
    rfpath = os.path.join(maindir, "model", "model_rf.pkl")
    svcpath = os.path.join(maindir, "model", "model_svc.pkl")

    model_knn = pickle.load(open(knnpath, 'rb'))
    model_rf = pickle.load(open(rfpath, 'rb'))
    model_svc = pickle.load(open(svcpath, 'rb'))

    #load test set
    xtestdf = pd.read_csv(xtestpath)
    ytestdf = pd.read_csv(ytestpath)


    #evaluate result
    #model knn
    ytest_prd_knn = model_knn.predict(xtestdf)
    acc_test_knn = accuracy_score(ytestdf, ytest_prd_knn)

    #model rf
    ytest_prd_rf = model_rf.predict(xtestdf)
    acc_test_rf = accuracy_score(ytestdf, ytest_prd_rf)

    #model svc
    ytest_prd_svc = model_svc.predict(xtestdf)
    acc_test_svc = accuracy_score(ytestdf, ytest_prd_svc)

    # ensure having result tab folder
    resDir = os.path.join(maindir, "result_tab")
    if not os.path.exists(resDir): 
        os.mkdir(resDir)

    #tabulate and output result in csv form
    rescsvpath = os.path.join(resDir, "result_table.csv")

    restab = {'Model': [ "KNN", "RF", "SVC"],
    'Accuracy Score': [acc_test_knn, acc_test_rf, acc_test_svc]
    }

    resdf = pd.DataFrame(restab)
    resdf.to_csv(rescsvpath, index=False, header = True)
    now = str(datetime.datetime.now())
    print(f"{now}: Done evaluation.")

    end = time.perf_counter()
    timeUsed = f"--- {end - start} seconds --- "
    print(timeUsed)


if __name__ == '__main__':
    # run('local')
    run('server')

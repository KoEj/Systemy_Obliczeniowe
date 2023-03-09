import threading
import numpy as np
import glob, os
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score

direct = "C:/Users/PLUSR6000280/PycharmProjects/SystemyObliczeniowe/csv/"


def getData(csvName):
    dataset = np.genfromtxt(csvName, delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    clf = GaussianNB()
    rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1234)
    scores = []

    for train_index, test_index in rkf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        scores.append(accuracy_score(y_test, predict))
    mean_score_clf = np.mean(scores)

    scores = []
    dtc = DecisionTreeClassifier()
    for train_index, test_index in rkf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dtc.fit(X_train, y_train)
        predict = dtc.predict(X_test)
        scores.append(accuracy_score(y_test, predict))
    mean_score_dtc = np.mean(scores)

    print (csvName, mean_score_clf, mean_score_dtc)
    return csvName, mean_score_clf, mean_score_dtc


threads = list()
fileList = ["balance.csv", "pima.csv"]
os.chdir(direct)
for item in glob.glob('*.csv'):
    x = threading.Thread(target=getData, args=(str(item),))
    threads.append(x)
    x.start()

for index, thread in enumerate(threads):
    print("Before joining thread %d", index)
    thread.join()
    print("Thread %d done", index)




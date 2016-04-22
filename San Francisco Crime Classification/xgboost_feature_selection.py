#encoding=utf-8
# This Program is written by Victor Zhang at 2016-02-01 14:37:28
#
#
import numpy as np
import pandas as pd
import xgboost as xgb
import csv
import time
from sklearn.metrics import log_loss
import operator
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder

sample = False
target='Category'


def load_data():
    if sample:
        train = pd.read_csv("./data/train_min.csv")
        test = pd.read_csv("./data/test_min.csv")
    else:
        train = pd.read_csv("./data/train.csv")
        test = pd.read_csv("./data/test.csv")
    features=['Dates','DayOfWeek','PdDistrict','Address','X','Y']
    return train,test,features

def findWayName(st):
    wayNames=[""," AV"," ST"," Block"," DR"," WY"," BL"," LN"," RD"," BLVD"," HY"," CT"," PZ"," TR"]
    for i,wayName in enumerate(wayNames):
        if i>0 and (wayName in st):
            return i
    return 0

def data_processing(train,test,features):
    print("Adding Features",time.ctime())
    # train['StreetNo'] = train['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
    # test['StreetNo'] = test['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)

    train['wayName']=train['Address'].apply(lambda x:findWayName(x))
    test['wayName']=test['Address'].apply(lambda x:findWayName(x))
    train['isInterSection']=train['Address'].apply(lambda x: 1 if '/' in x else 0)
    test['isInterSection']=test['Address'].apply(lambda x: 1 if '/' in x else 0)
    train['Address'] = train['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
    test['Address'] = test['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)

    train['hour']= train['Dates'].apply(lambda x: x[11:13] if len(x) > 4 else 12)
    test['hour'] = test['Dates'].apply(lambda x: x[11:13] if len(x) > 4 else 12)
    train['dark'] = train['Dates'].apply(lambda x: 1 if (len(x) > 4 and (int(x[11:13]) >= 18 or int(x[11:13]) < 6)) else 0)
    test['dark'] = test['Dates'].apply(lambda x: 1 if (len(x) > 4 and (int(x[11:13]) >= 18 or int(x[11:13]) < 6)) else 0)

    features += ['wayName','isInterSection','hour','dark']

    # print(train[['Address','StreetNo']])

    print("Filling NAs",time.ctime())
    # print(train.mode())
    train = train.fillna(train.median().iloc[0])
    test = test.fillna(test.median().iloc[0])
    print("Label Encoder",time.ctime())
    le=LabelEncoder()
    for col in features:
        le.fit(list(train[col])+list(test[col]))
        train[col]=le.transform(train[col])
        test[col]=le.transform(test[col])

    le.fit(list(train[target]))
    train[target]=le.transform(train[target])

    print("Standard Scalaer",time.ctime())
    scaler=StandardScaler()
    for col in features:
        scaler.fit(list(train[col]))
        train[col]=scaler.transform(train[col])
        test[col]=scaler.transform(test[col])

    return train,test,features

def XG_boost(train,test,features):
    params = {'max_depth':8, 'eta':0.05, 'silent':1,
              'objective':'multi:softprob', 'num_class':39, 'eval_metric':'mlogloss',
              'min_child_weight':3, 'subsample':0.6,'colsample_bytree':0.6, 'nthread':4}
    num_rounds = 250


    print("Start Training",time.ctime())
    xgbtrain = xgb.DMatrix(train[features], label=train[target])
    classifier = xgb.train(params, xgbtrain, num_rounds)
    print("Start Predicting",time.ctime())
    dtest=xgb.DMatrix(test[features])
    ans=classifier.predict(dtest)
    ansSize=ans.shape[0]

    csvfile = 'result/xgboost-feature-submit.csv'
    with open(csvfile, 'w') as output:
        predictions = []

        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(['Id','ARSON','ASSAULT','BAD CHECKS','BRIBERY','BURGLARY','DISORDERLY CONDUCT',
                         'DRIVING UNDER THE INFLUENCE','DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT',
                         'EXTORTION','FAMILY OFFENSES','FORGERY/COUNTERFEITING','FRAUD','GAMBLING',
                         'KIDNAPPING','LARCENY/THEFT','LIQUOR LAWS','LOITERING','MISSING PERSON',
                         'NON-CRIMINAL','OTHER OFFENSES','PORNOGRAPHY/OBSCENE MAT','PROSTITUTION',
                         'RECOVERED VEHICLE','ROBBERY','RUNAWAY','SECONDARY CODES','SEX OFFENSES FORCIBLE',
                         'SEX OFFENSES NON FORCIBLE','STOLEN PROPERTY','SUICIDE','SUSPICIOUS OCC','TREA',
                         'TRESPASS','VANDALISM','VEHICLE THEFT','WARRANTS','WEAPON LAWS'])

        for i in range(ansSize):
            # import pdb;pdb.set_trace()
            predictions += [[i]+ans[i].tolist()]
            if i%50000==0:
                writer.writerows(predictions)
                predictions=[]
        writer.writerows(predictions)
        print("Predicting done",time.ctime())

    outfile = open('result/xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()
    importance = classifier.get_fscore(fmap='result/xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df.to_csv('result/importance.csv',index=False)


if __name__ == '__main__':
    train,test,features=load_data()
    train,test,features=data_processing(train,test,features)
    XG_boost(train,test,features)

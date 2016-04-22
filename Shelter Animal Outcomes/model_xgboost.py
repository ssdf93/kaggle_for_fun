#encoding=utf-8
# This Program is written by Victor Zhang at 2016-04-22 19:47:39
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


sample = True
target='OutcomeType'


def load_data():
    if sample:
        train = pd.read_csv("./data/train_min.csv")
        test = pd.read_csv("./data/test_min.csv")
    else:
        train = pd.read_csv("./data/train.csv")
        test = pd.read_csv("./data/test.csv")

    return train,test

def birthAge(x):
    sts=x.split(' ')
    if 'day' in x:
        return int(sts[0])
    elif 'week' in x:
        return int(sts[0])+10
    elif 'month' in x:
        return int(sts[0])+30
    elif 'year' in x:
        return int(sts[0])+50
    else:
        return None


def data_processing(train,test):
    features=[]
    for data in [train,test]:
        data['AgeuponOutcome']=data['AgeuponOutcome'].fillna('Unknown')
        data['birthAge']=data['AgeuponOutcome'].apply(lambda x: birthAge(x))


    # print(data['birthAge'][:10])


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




def write_csv(file_name,ans,first_row=None,myId=None):
    """
    Write the data to csv ,
    """
    # csvfile = 'results/xgboost-feature-submit.csv'

    ansSize=ans.shape[0]
    with open(file_name, 'w') as output:
        predictions = []

        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(first_row)

        for i in range(ansSize):
            # import pdb;pdb.set_trace()
            predictions += [list(myId[i])+ans[i].tolist()]
            if i%50000==0:
                writer.writerows(predictions)
                predictions=[]
        if predictions != None:
            writer.writerows(predictions)

        print("Writing CSV done",time.ctime())

    # outfile = open('result/xgb.fmap', 'w')
    # i = 0
    # for feat in features:
    #     outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    #     i = i + 1
    # outfile.close()
    # importance = classifier.get_fscore(fmap='result/xgb.fmap')
    # importance = sorted(importance.items(), key=operator.itemgetter(1))
    # df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    # df.to_csv('result/importance.csv',index=False)


if __name__ == '__main__':
    train,test=load_data()
    train,test,features=data_processing(train,test)

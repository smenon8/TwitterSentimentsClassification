import argparse
import sklearn.feature_extraction.text as txt
from sklearn import svm,naive_bayes,tree
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,roc_curve
from sklearn.model_selection import cross_val_predict, cross_val_score
import csv
import pandas as pd
import re
import json
import numpy as np
from collections import Counter, OrderedDict
import sys
import time

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Conversion to bag-of-words (binary model)
def gen_BagOfWords(recds,allAttribs,header='Anootated tweet'):
    data = []
    for rec in recds:
        if len(rec[header]) > 0:
            ftrVector = dict.fromkeys(allAttribs,0)
            for word in rec[header]:
                if word != '' and word in allAttribs:
                    ftrVector[word] = ftrVector.get(word,0) + 1
                ftrVector['class'] = rec['Class']
            data.append(ftrVector)

    df = pd.DataFrame(data)
    
    return df

# TF-IDF vectorizer
# Returns only the features and the target class
def tfIDfVectorizer(df):
    y = df['class']

    if '' in df.columns:
        df.drop(['class',''],1,inplace=True)
    else:
        df.drop(['class'],1,inplace=True)

    trfm = txt.TfidfTransformer()
    trfm.fit(df)
    matrx = trfm.transform(df)
    matrx = matrx.todense()
    return matrx.A, y

# Generates all the words used in the tweets provided.
def genVocab(dictFile,header='Anootated tweet'):
    allFtrs = []
    for row in dictFile:
        allFtrs.extend(row[header])
    
    allFtrCnt = Counter(allFtrs)
    cntSorted = OrderedDict()
    
    attribs = sorted(allFtrCnt.keys(), key = lambda x : allFtrCnt[x], reverse=True)
    for attrib in attribs:
        if attrib != 'class':
            cntSorted[attrib] = allFtrCnt[attrib]
    
    return list(cntSorted.keys())

def dataClean(flNm = "../data/training-Obama-Romney-tweets.xlsx"):
    df_obama = pd.read_excel(flNm,sheetname=0).dropna(how='any')
    df_romney = pd.read_excel(flNm,sheetname=1).dropna(how='any')

    cleanTagsRE = re.compile('<.*?>')
    cleanFnc = lambda x : re.sub(cleanTagsRE, '', x)
    remSymRE = re.compile('[^a-zA-Z0-9]')
    remSymFnc = lambda x : remSymRE.sub('',x)

    df_obama['Anootated tweet'] = df_obama['Anootated tweet'].apply(cleanFnc)
    df_romney['Anootated tweet'] = df_romney['Anootated tweet'].apply(cleanFnc)

    # Ignore class 2 - mixed class
    df_obama = df_obama[(df_obama['Class'].isin((1,-1,0)))]
    df_romney = df_romney[(df_romney['Class'].isin((1,-1,0)))]

    obama_rec = df_obama.to_dict(orient='records')
    romney_rec = df_romney.to_dict(orient='records')

    for tweet in obama_rec:
        tweet['Anootated tweet'] = list(map(remSymFnc,tweet['Anootated tweet'].split()))
        tweet['Anootated tweet'] = [ele.lower() for ele in tweet['Anootated tweet'] if not 'http' in ele and not ele.isdigit()]

    for tweet in romney_rec:
        tweet['Anootated tweet'] = list(map(remSymFnc,tweet['Anootated tweet'].split()))
        tweet['Anootated tweet'] = [ele.lower() for ele in tweet['Anootated tweet'] if not 'http' in ele and not ele.isdigit()]

    # stop words removal using nltk library
    stopWords = set(stopwords.words("english"))
    for tweet in obama_rec:
        tweet['Anootated tweet'] = list(filter(lambda x: x not in stopWords,tweet['Anootated tweet']))

    for tweet in romney_rec:
        tweet['Anootated tweet'] = list(filter(lambda x: x not in stopWords,tweet['Anootated tweet']))


    # stemming using nltk library
    p = PorterStemmer()
    for tweet in obama_rec:
        tweet['Anootated tweet'] = list(map(lambda x: p.stem(x),tweet['Anootated tweet']))

    for tweet in romney_rec:
        tweet['Anootated tweet'] = list(map(lambda x: p.stem(x),tweet['Anootated tweet']))

    if 'training' in flNm:
        outFlObama = "../data/obamaTweets.json"
        outFlRomney = "../data/romneyTweets.json"
    else:
        outFlObama = "../data/obamaTweets_test.json"
        outFlRomney = "../data/romneyTweets_test.json"

    with open(outFlObama,"w") as obamaJson:
        json.dump(obama_rec,obamaJson,indent=4)
        
    with open(outFlRomney,"w") as romneyJson:
        json.dump(romney_rec,romneyJson,indent=4)

    print("Data clean complete.")
    return None

def feature_gen(flNm):
    with open(flNm,"r") as candidJson:
        candid_rec = json.load(candidJson)       

    allAttribs= genVocab(candid_rec)

    # Extremely expensive step. Avoid re-running.
    df = gen_BagOfWords(candid_rec,allAttribs)
    X, y = tfIDfVectorizer(df)

    return X, y, allAttribs

def getClf(methodName):
    if methodName == 'logistic':
        return LogisticRegression()
    elif methodName == 'svm':
        n_estimators = 10
        return OneVsRestClassifier(BaggingClassifier(svm.SVC(kernel='linear'), max_samples = 1.0/ n_estimators, n_estimators = n_estimators, n_jobs = 10))
    elif methodName == 'dtree':
        return tree.DecisionTreeClassifier(criterion='entropy')
    elif methodName == 'rf':
        return RandomForestClassifier(criterion='entropy', n_jobs = 10)
    elif methodName == 'bayes':
        return naive_bayes.BernoulliNB()
    elif methodName == 'ada_boost':
        return AdaBoostClassifier()
    else:
        try:
            raise Exception('Exception : Classifier Method %s Unknown' %methodName)
        except Exception as inst:
            print(inst.args)
            sys.exit()

def train(clfType, X, y):
    clf = getClf(clfType)

    start = time.time()
    clf.fit(X, y)
    end = time.time()

    print("Total training time: %0.3f s" %(end-start))

    return clf

def test(clfObj, allAttribs, testFl):
    with open(testFl,"r") as candidJson:
        candid_rec = json.load(candidJson)

    df = gen_BagOfWords(candid_rec, allAttribs)
    df.fillna(0.0,inplace=True)
    test_x, test_y = tfIDfVectorizer(df)

    preds = clfObj.predict(test_x)

    accScore = accuracy_score(test_y, preds)
    
    labels = [1,-1,0]
    precision = precision_score(test_y, preds,average=None, labels=labels)
    recall = recall_score(test_y, preds, average=None, labels=labels)
    f1Score = f1_score(test_y, preds, average=None, labels=labels)

    print("Testing accuracy")
    print("Overall Acurracy : ",accScore)
    lbl = ['positive', 'negative', 'nuetral']
    for i in range(3):
        print("Precision of %s class: %f" %(lbl[i],precision[i]))
        print("Recall of %s class: %f" %(lbl[i],recall[i]))
        print("F1-Score of %s class: %f" %(lbl[i],f1Score[i]))
        print()

def classify_CV(clfType, X, y):
    clf = getClf(clfType)

    start = time.time()
    preds = cross_val_predict(clf, X, y, cv=10)
    end = time.time()

    print("Total training & 10 fold cross validation time: %0.3f s" %(end-start))

    det_acc_score = cross_val_score(clf, X, y, cv=10)

    for i in range(0,10):
        print("Iteration %d: score = %f" %(i+1,det_acc_score[i]))
    print()

    accScore = accuracy_score(y,preds)
    
    labels = [1,-1,0]
    precision = precision_score(y,preds,average=None,labels=labels)
    recall = recall_score(y,preds,average=None,labels=labels)
    f1Score = f1_score(y,preds,average=None,labels=labels)

    print("Overall Acurracy : ",accScore)
    print("Precision : ",precision)
    print("Recall : ",recall)
    print("F1-Score : ",f1Score)


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dc','--clean', help='Use this for data cleaning, specify excel file name.', required=False)
    parser.add_argument('-c','--candidate', help='Specify candidate.', required=True)
    parser.add_argument('-tr','--train', help='Specify True if specified', required=False)
    parser.add_argument('-te','--test', help='Specify True if specified', required=False)
    parser.add_argument('-cv','--cross_validate', help='Use this for classifier training and cross validation', required=False)
    parser.add_argument('-clf','--classifier', help='Specificy classification algorithm to be used', required=False)

    args = vars(parser.parse_args())

    if args["clean"] != None:
        print("Starting data cleaning steps")
        dataClean(args["clean"])
    else:
        if args['classifier'] != None and args["candidate"] != None:
            if args['classifier'] in ['svm','bayes','dtree','rf','logistic','ada_boost']:
                clfNm = args['classifier']
            else:
                print("Unknown classifier")
                sys.exit(-1)

        if args["candidate"] in ['obama', 'romney']: 
            candidate = args["candidate"]
        else:
            print("Acceptable candidate names are Obama, Romney")
            sys.exit(-1)
    
        if args["cross_validate"] != None:
            if candidate == 'obama':
                flNm = "../data/obamaTweets.json"
            else:
                flNm = "../data/romneyTweets.json"
            print("Starting feature generation steps __main__")
            X, y, _ = feature_gen(flNm)
            print("Feature generation complete __main__\n")

            print("Cross validation started __main__")
            classify_CV(clfNm, X, y)
            print("Cross validation completed __main__\n")
        else:
            if args["train"] != None and args["test"] != None:
                if candidate == 'obama':
                    flNm_train = "../data/obamaTweets.json"
                    flNm_test = "../data/obamaTweets_test.json"
                else:
                    flNm_train = "../data/romneyTweets.json"
                    flNm_test = "../data/romneyTweets_test.json"

                print("Starting feature generation steps __main__")
                train_x, train_y, all_attribs = feature_gen(flNm_train)
                print("Feature generation complete __main__\n")

                print("Classifier training started __main__")
                clf_obj = train(clfNm, train_x, train_y)
                print("Classifier training complete __main__\n")

                print("Testing on data started __main__")
                test(clf_obj, all_attribs, flNm_test)
                print("Testing on data completed __main__")
            else:
                print("Training or Testing data required if --cross_validate option is unset.")


if __name__ == "__main__":
     __main__()
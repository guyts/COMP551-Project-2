# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:02:08 2017
@author: gtsror
"""

import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import KFold
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup             
import re

        
def sanitizer(text):
    # this function cleans up reddit texts from HTML characters, punctuations
    # and stopwords, as well as uncapitalizes all words.
    
    # cleaning text from <> and other html chars
    tmp = BeautifulSoup(text)  
    # removing punctuations etc, capitals, and splitting into words
    words = (re.sub("[^a-zA-Z]", " ", tmp.get_text())).lower().split()
    # removing stop words as defined in nltk
    meaningfll = [w for w in words if not w in stop_words]
    lst=[]    
    for j in meaningfll:
        tmp3 = lemmatizer.lemmatize(j,pos='v')
        lst.append(tmp3)
        lst.append(" ")
    cleanTxt = "".join(lst)
    return cleanTxt


def sigFn(weights,x):
    # the sigmoid function, where z = weightsTx (weights multipled by the features)
    return 1 / (1 + np.exp(-1*np.dot(np.transpose(weights),x)))


def gradDesc2(X,Y,weights,alpha):
    
    # using the mini batch gradient descent we'll randomly choose 100 instances
    # and find the descent on them
#    randInd = random.sample(range(np.size(Y)), 100)
    X1 = X#[randInd]
    Y1 = Y#[randInd]
    weights2 = []
    diffind = np.zeros(np.size(Y1))
    for i in range(0,np.size(diffind)):
        ind = i#randInd[i]
        diffind[i] = sigFn(weights,X1[ind])-Y1[ind]
    gradD = np.dot(diffind,X1)
    norm2 = np.linalg.norm(gradD)
#    print("Current gradient norm is: %f" % norm2)
    weights2 = weights - alpha*gradD
    
    return weights2, norm2  #returning the new vector of weights
    

def trainLogRegModel(X_train, Y_train, alpha):

    weights = np.zeros(np.size(X_train,1))
    weightsTmp = np.copy(weights)
    norm = [0]
    for i in range(0,200000):
#        startTime = time.time()
        weightsTmp, normTmp = gradDesc2(X_train,Y_train,weightsTmp,alpha)
        weights = np.column_stack((weights,weightsTmp))
        norm = np.column_stack((norm,normTmp))
        if i>0:
#            print(" Mean difference between weights: %f\n" % np.mean((np.around(weightsTmp, decimals=5)-np.around(weights[:,i-1], decimals=5))))
#            if np.array_equal(np.around(weightsTmp, decimals=5), np.around(weights[:,i-1], decimals=5)):
#            if i%100==0:
#                print("Run number: %d, gradient norm is: %f" % ((i+1), normTmp))

            if norm[0,i]<np.size(X,1)/2:
#                print("**********************\nStopped after: %d iterations" % i)
                break
#        print("runtime %i round is: %f seconds" % (i+1, time.time()-startTime))
    # now that gradient descent is over, we have the set of weights in weights[:,last]
    # those weights will
    return weights

def kFoldSplit(X,Y,nFolds,runNum):
    # this function creates a training set and a validation set from X and Y,...
    # based on the number of folds desired.
    # runNum is the number of time that the Kfold is running (from 1 to nFolds)
    curRun=0
    kf = KFold(np.size(Y),nFolds)
    for train_index, val_index in kf:
        curRun = curRun+1
        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]
        if curRun == runNum: # if we are in the kth run, we want this division of data
            return X_train, X_val, Y_train, Y_val

        
df = pd.read_csv("C:/Users/guyts/OneDrive/OneDrive/Important Docs/School MSc/McGill/Semester B/COMP551/Projects/Project 2/train_input.csv")
df2 = pd.read_csv("C:/Users/guyts/OneDrive/OneDrive/Important Docs/School MSc/McGill/Semester B/COMP551/Projects/Project 2/train_output.csv")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

convos = df['conversation']
numConvos = np.size(convos)

clean_training_data = [[] for j in range(0,numConvos)]

for i in range(0,numConvos):
    clean_training_data[i] =   sanitizer(df['conversation'][i])   
    if( (i+1)%1000 == 0 ):
        print("Checked %d of %d\n" % ( i+1, numConvos ))

# defining the output training vector y. Changing to ints from strs           
yCategorical = np.zeros([np.size(df2,0), 1],dtype=np.int)
for i in range(0,np.size(df2,0)):
    if df2['category'][i]=='news':
        yCategorical[i] = 0
    elif df2['category'][i]=='nfl':
        yCategorical[i] = 1
    elif df2['category'][i]=='soccer':
        yCategorical[i] = 2
    elif df2['category'][i]=='movies':
        yCategorical[i] = 3
    elif df2['category'][i]=='politics':
        yCategorical[i] = 4
    elif df2['category'][i]=='hockey':
        yCategorical[i] = 5
    elif df2['category'][i]=='nba':
        yCategorical[i] = 6
    elif df2['category'][i]=='worldnews':
        yCategorical[i] = 7
Y = np.copy(yCategorical)

# Initializing the  bagofwords tool:
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 4000) 

feats = vectorizer.fit_transform(clean_training_data)
#TFIDF
tf_transformer = TfidfTransformer(use_idf=True).fit(feats)
X = tf_transformer.transform(feats)
Xold = X.toarray() #TF IDF
#Xold = feats.toarray() #Bag of Words

# defining a smaller data set for quicker runtime 
# N/10
N2 = int(len(Y)/10)

Ynew = np.copy(Y[0:N2]);
del Y
Y = Ynew
X = np.copy(Xold[0:N2,:])
#vocab = vectorizer.get_feature_names() #all the features



        
# setting up the multiple variables code 

# Category 0 need sto be finished by finding the validation prediction)
# once thats done, the second category must be ran and so on
# after all categories ran for the first fold, now we need to find the actual 
# prediction, by choosing the categories with the maxiaml hProb
# Then, the loop moves on to the next fold.
# eventually, calculate the error after all folds are done.

nFolds = 5
correct = np.zeros(nFolds)
for kfld in range(0,nFolds):
    startT=time.time()
    hProb = np.zeros((np.size(Y)/nFolds,8))

    for category in range(0,8):

        X_train, X_val, Y_train, Y_val = kFoldSplit(X,Y,nFolds,kfld+1)
        Y_est_val = np.zeros(np.size(Y_val))
        # first case: identifying news
        if category == 0:
            # train on training data\
            Ycur0 = np.copy(Y_train)
            # defining 1 where the category is right, and 0 where not
            Ycur0[Y_train==category] = 1
            Ycur0[Y_train!=category] = 0
            # find weights and hypothesis matrix (using sigFn)
            weights0 = trainLogRegModel(X_train, Ycur0, 1/55000)
            #753 iterations with a=1/8, 13200 training points, 500 features
            
                # using the weights found here, we can find now the probability of each
                # document to be from this class
                
            for doc in range(0,np.size(Y_val)):
                    # running and calculating the hypothesis (hProb) for each doc, to generate
                    # the column vector hProb for this category. if that hypothesis 
                    # is larger than 0.5 - we decide it's part of this category. 
                    hProb[doc,category] = sigFn(weights0[:,-1],X_val[doc,:])
#                    if hProb[doc,category] > 0.5:
#                    # if this probablillity is >0.5, for now we'll assume we can classify
#                    # it into the current cateogry
#                        Y_est_val[doc] = 1
            print("****************\nCategory %d done!" % category)

                
            # hprob returns from grad descent
        # 2nd case: identifying nfl
        elif category == 1:
            Ycur1 = np.copy(Y_train)
            Ycur1[Y_train==category] = 1
            Ycur1[Y_train!=category] = 0
            weights1 = trainLogRegModel(X_train, Ycur1, 1/55000)
            #939 iterations with a=1/8, 13200 training points, 500 features

            for doc in range(0,np.size(Y_val)):
                    hProb[doc,category] = sigFn(weights1[:,-1],X_val[doc,:])
            print("****************\nCategory %d done!" % category)
        # 3rd case: identifying soccer
        elif category == 2:    
            Ycur2 = np.copy(Y_train)
            Ycur2[Y_train==category] = 1
            Ycur2[Y_train!=category] = 0
            weights2 = trainLogRegModel(X_train, Ycur2, 1/55000)
            for doc in range(0,np.size(Y_val)):
                    hProb[doc,category] = sigFn(weights2[:,-1],X_val[doc,:])
            print("****************\nCategory %d done!" % category)
        # case 4: identifying movies
        elif category == 3: 
            Ycur3 = np.copy(Y_train)
            Ycur3[Y_train==category] = 1
            Ycur3[Y_train!=category] = 0
            weights3 = trainLogRegModel(X_train, Ycur3, 1/55000)
            for doc in range(0,np.size(Y_val)):
                    hProb[doc,category] = sigFn(weights3[:,-1],X_val[doc,:])
            print("****************\nCategory %d done!" % category)
        # case 5: identifying politics
        elif category == 4:  
            Ycur4 = np.copy(Y_train)
            Ycur4[Y_train==category] = 1
            Ycur4[Y_train!=category] = 0
            weights4 = trainLogRegModel(X_train, Ycur4, 1/55000)
            for doc in range(0,np.size(Y_val)):
                    hProb[doc,category] = sigFn(weights4[:,-1],X_val[doc,:])
                    
            print("****************\nCategory %d done!" % category)
        # case 6: identifying hockey
        elif category == 5:  
            Ycur5 = np.copy(Y_train)
            Ycur5[Y_train==category] = 1
            Ycur5[Y_train!=category] = 0
            weights5 = trainLogRegModel(X_train, Ycur5, 1/55000)
            
            for doc in range(0,np.size(Y_val)):
                    hProb[doc,category] = sigFn(weights5[:,-1],X_val[doc,:])
            print("****************\nCategory %d done!" % category)
        # case 7: identifying nba
        elif category == 6:  
            Ycur6 = np.copy(Y_train)
            Ycur6[Y_train==category] = 1
            Ycur6[Y_train!=category] = 0
            weights6 = trainLogRegModel(X_train, Ycur6, 1/55000)
            
            for doc in range(0,np.size(Y_val)):
                    hProb[doc,category] = sigFn(weights6[:,-1],X_val[doc,:])
            print("****************\nCategory %d done!" % category)
        # case 8: identifying worldnews
        elif category == 7:  
            Ycur7 = np.copy(Y_train)
            Ycur7[Y_train==category] = 1
            Ycur7[Y_train!=category] = 0
            weights7 = trainLogRegModel(X_train, Ycur7, 1/55000)
            
            for doc in range(0,np.size(Y_val)):
                    hProb[doc,category] = sigFn(weights7[:,-1],X_val[doc,:])
            print("****************\nCategory %d done!" % category)
    
    classes = np.argmax(hProb, axis=1) # gives back the ind for the maximal value of each row
    tp=0
    TN=0
    FP=0
    FN=0
    for i in range(0,len(classes)):
        if classes[i] == Y_val[i]:
            tp = tp+1
    correct[kfld] = tp/len(classes)
    print("Runtime for kfld #%d: %f " % (kfld, float(time.time()-startT)))
#now we can say that the final classified values are the ones given by classes
avePercent = np.mean(correct)
print(correct[3])


# Evaluating the results using the classification report

target_names = ['news', 'nfl', 'soccer','movies','politics','hockey','nba','worldnews']
print(classification_report(Y_val, classes, target_names=target_names))

# -*- coding: utf-8 -*-
#kaggle titanic disaster competition
from scipy import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings
import time


warnings.filterwarnings('ignore')

STARTTIME = time.time()

#data directory. Change as necessary!
directory = "/Users/yumi/Downloads/titanic/"

#function list:
def getHonorific(DF,i):
    #get the honorific from Name
    return DF["Name"].loc[i].split(",")[1].split(".")[0].strip()

def loadData(directory):
    TRAIN = pd.read_csv(directory+"train.csv")
    TEST = pd.read_csv(directory+"test.csv")
    return TRAIN, TEST
    
def extractHonorifics(TRAIN,TEST,NOB=0):
    #we simplify the entire honorific set to a simplified version of just
    #Master, Ms, Mr and Mrs
    honorifics=[]
    for i in range(len(TRAIN["Name"])):
        honorifics.append(getHonorific(TRAIN,i))
    for i in range(len(TEST["Name"])):
        honorifics.append(getHonorific(TEST,i))  
    if NOB==0:
        toMr=["Capt","Col","Don","Dr","Jonkheer","Major","Rev","Sir"]
        toMrs=["Dona","Lady","Mme","the Countess"]
        toMs=["Mlle","Miss"]
        for i in range(len(honorifics)):
            if(honorifics[i] in toMr):
                honorifics[i] = "Mr"
            elif(honorifics[i] in toMrs):
                honorifics[i] = "Mrs"
            elif(honorifics[i] in toMs):
                honorifics[i] = "Ms"
    elif NOB>0:
        toNob=["the Countess","Dona","Lady","Capt","Col","Don","Dr","Jonkheer","Major","Rev","Sir"]
        toMrs=["Mme"]
        toMs=["Mlle","Miss"]
        for i in range(len(honorifics)):
            if(honorifics[i] in toNob):
                honorifics[i] = "Nobility"
            elif(honorifics[i] in toMrs):
                honorifics[i] = "Mrs"
            elif(honorifics[i] in toMs):
                honorifics[i] = "Ms"        
    return honorifics
    
def scatterPlot(X,feature1,feature2):
    plt.scatter(X[feature1],X[feature2])  
    plt.xlabel(feature1)
    plt.ylabel2(feature2)
    plt.show()  

def checkForNan(DF,COLUMN):
    X = pd.isnull(DF[COLUMN])
    return X.sum()
         
def binColumn(TRAIN,TEST,COL,N):   
    #this function bins the data in a particular column COL of both the train &
    #test dataframes into N+1 different bins.
    nanflag=pd.isnull(TRAIN[COL]).sum()+pd.isnull(TEST[COL]).sum()   
    maxVal = int(ceil(max([TRAIN[COL].max(),TEST[COL].max()])))
    #in the case where the float max value is equal to the int max value,
    #add 1 to the int max val so that the floats will be binned properly!
    if max([TRAIN[COL].max(),TEST[COL].max()]) == maxVal:
        maxVal=maxVal+1
    minVal = int(floor(min([TRAIN[COL].min(),TEST[COL].min()])))
    dC = (maxVal - minVal)/N
    bins = range(minVal,maxVal+dC,dC)
    labels = range(1,len(bins),1)
    new_col = "Binned_"+COL.lower()
    if nanflag==0:    
        #this creates problems if NaN are present in the data
        TRAIN[new_col]=pd.cut(TRAIN[COL], bins=bins, labels=labels,right=False)
        TEST[new_col]=pd.cut(TEST[COL], bins=bins, labels=labels,right=False)   
    elif nanflag>0:
        print "Warning! NaN detected!"   
        new_col_train = pd.cut(TRAIN[COL][pd.notnull(TRAIN[COL])],bins=bins,labels=labels,right=False)
        new_col_test = pd.cut(TEST[COL][pd.notnull(TEST[COL])],bins=bins,labels=labels,right=False)
        TRAIN[new_col]=nan
        TEST[new_col]=nan
        TRAIN[new_col][pd.notnull(TRAIN[COL])]=new_col_train
        TEST[new_col][pd.notnull(TEST[COL])]=new_col_test
    return TRAIN,TEST,bins,labels

def binColumnCustom(TRAIN,TEST,COL,BINS,WRITEOVER=False):
    #unlike the previous function, this one bins the data in COL of train and
    #test using the provided BINS
    TRAIN["Binned_"+COL.lower()]=nan
    TEST["Binned_"+COL.lower()]=nan
    for i in range(1,1+len(BINS[1:])):
        LOWERLIM=BINS[i-1]
        UPPERLIM=BINS[i]
        TRAIN.loc[(TRAIN[COL]<UPPERLIM)&(TRAIN[COL]>=LOWERLIM),"Binned_"+COL.lower()]=i
        if WRITEOVER:
            TRAIN.loc[(TRAIN[COL]<UPPERLIM)&(TRAIN[COL]>=LOWERLIM),COL]=(UPPERLIM+LOWERLIM)/2
    for i in range(1,1+len(BINS[1:])):
        LOWERLIM=BINS[i-1]
        UPPERLIM=BINS[i]
        TEST.loc[(TEST[COL]<UPPERLIM)&(TEST[COL]>=LOWERLIM),"Binned_"+COL.lower()]=i
        if WRITEOVER:
            TEST.loc[(TEST[COL]<UPPERLIM)&(TEST[COL]>=LOWERLIM),COL]=(UPPERLIM+LOWERLIM)/2
    return TRAIN,TEST 

#end function list
      
#load the necessary data          
TRAIN,TEST = loadData(directory)

#Feature engineering:

#first we simplify the entire honorifics set to a more simplified set
honorifics = extractHonorifics(TRAIN,TEST,NOB=1)
print "Reduced honorifics set:"
for i in unique(honorifics):
    print "{}:{}".format(i,honorifics.count(i))
TRAIN["Honorific"] = pd.DataFrame(honorifics[:len(TRAIN)])
TEST["Honorific"] = pd.DataFrame(honorifics[len(TRAIN):])

#calculate the number of travelling companions for each passenger
TRAIN["Companions"] = TRAIN["SibSp"] + TRAIN["Parch"]
TEST["Companions"] = TEST["SibSp"] + TEST["Parch"]

#also determine if the passenger is alone
TRAIN["Is_alone"] = 0
TEST["Is_alone"] = 0
TRAIN.loc[TRAIN['Companions'] == 0, 'Is_alone'] = 1
TEST.loc[TEST['Companions'] == 0, 'Is_alone'] = 1

CABINS = []
for i in range(len(TRAIN.Cabin)):
    CABINS.append(TRAIN.Cabin[i])
for i in range(len(TEST.Cabin)):
    CABINS.append(TEST.Cabin[i])

for i in range(len(CABINS)):
    if(not(pd.isnull(CABINS[i]))):
        CABINS[i] = CABINS[i][0]
    else:
        #replace NaN with N
        CABINS[i] = "N"

print "Unique cabin letters:"
for i in unique(CABINS):
    print "{}:{}".format(i,CABINS.count(i))

TRAIN["Cabin_letter"] = pd.DataFrame(CABINS[:len(TRAIN.Cabin)])
TEST["Cabin_letter"] = pd.DataFrame(CABINS[len(TRAIN.Cabin):])

#cabins starting with G and T are too few in number. Perhaps we should lump
#them with other cabins, or simply group people into "inside cabin" and
#"not inside cabin"
TRAIN["In_cabin"] =  TRAIN["Cabin"].notnull()
TEST["In_cabin"] =  TEST["Cabin"].notnull()
#directly convert True/False to 1/0
TRAIN["In_cabin"]=TRAIN["In_cabin"].replace({True:1,False:0})
TEST["In_cabin"]=TEST["In_cabin"].replace({True:1,False:0})

#also convert Sex from Male/Female to 1/0
TRAIN["Sex"]=TRAIN["Sex"].replace({"male":1,"female":0})
TEST["Sex"]=TEST["Sex"].replace({"male":1,"female":0})

#also convert Honorific from Master, Mr, Ms, Mrs, Nobility to 0,1,2,3,4
TRAIN["Honorific"]=TRAIN["Honorific"].replace({"Master":0,"Mr":1,"Ms":2,"Mrs":3,"Nobility":4})
TEST["Honorific"]=TEST["Honorific"].replace({"Master":0,"Mr":1,"Ms":2,"Mrs":3,"Nobility":4})

print "Number of null values per column in TRAIN:"
for i in TRAIN.columns:
    if checkForNan(TRAIN,i) > 0:
        print "{}:{}".format(i,checkForNan(TRAIN,i))
print "Number of null values per column in TEST:"
for i in TEST.columns:
    if checkForNan(TEST,i) > 0:
        print "{}:{}".format(i,checkForNan(TEST,i))        

#fill NaN in Cabin with N
TRAIN['Cabin']= TRAIN['Cabin'].fillna("N")
TEST['Cabin']= TEST['Cabin'].fillna("N")

#we have two null values in TRAIN.Embarked. Replace them with the mode()
embarkedMode = TRAIN["Embarked"].mode()
TRAIN["Embarked"] = TRAIN["Embarked"].fillna(embarkedMode.values[0])
#also replace the str values with int values: {"C":0,"Q":1,"S":2}
TRAIN["Embarked"]=TRAIN["Embarked"].replace({"C":0,"Q":1,"S":2})
TEST["Embarked"]=TEST["Embarked"].replace({"C":0,"Q":1,"S":2})

#we have a single null value in TEST.Fare. Replace with the mode()
fareMode = TEST["Fare"].mode()
TEST["Fare"] = TEST["Fare"].fillna(fareMode.values[0])

#bin fares using a specific cut
fareBins = array([0,8,15,32,999])
#fareBins = arange(0,513+27,27)
TRAIN,TEST=binColumnCustom(TRAIN,TEST,"Fare",fareBins)

#bin fares using a uniform cut
#TRAIN,TEST,fareBins,fareLabels = binColumn(TRAIN,TEST,"Fare",20)
print "Binned fare value_counts():"
print TRAIN.Binned_fare.value_counts()
print TEST.Binned_fare.value_counts()

#bin ages using a specific cut
#ageBins = array([0,16,32,48,64,99])
#ageBins = array([0,20,45,100])
ageBins = array([0,16,32,48,64,80,99])
TRAIN,TEST=binColumnCustom(TRAIN,TEST,"Age",ageBins)

#bin ages using a uniform cut
#TRAIN,TEST,ageBins,ageLabels = binColumn(TRAIN,TEST,"Age",5)
#print "Binned age value_counts():"
#print TRAIN.Binned_age.value_counts()
#print TEST.Binned_age.value_counts()

#now we need to deal with the overwhelmingly large number of null values in Age
#we will attempt to predict the missing age values using AI...
#also we will attempt tp predict the age bins rather than the exact age
XAGE = pd.concat([TRAIN.drop("Survived",axis=1),TEST],ignore_index=True)
#drop features unnecessary for fitting
XAGE.drop("PassengerId",axis=1,inplace=True)
XAGE.drop("Name",axis=1,inplace=True)
XAGE.drop("Cabin",axis=1,inplace=True)
XAGE.drop("Ticket",axis=1,inplace=True)
XAGE.drop("Cabin_letter",axis=1,inplace=True)
XAGE.drop("Sex",axis=1,inplace=True)
XAGE.drop("Age",axis=1,inplace=True)
yAGE=XAGE["Binned_age"]

XAGE_train = XAGE.iloc[list(pd.notnull(XAGE["Binned_age"]))]
XAGE_test = XAGE.iloc[list(pd.isnull(XAGE["Binned_age"]))]
yAGE_train = yAGE.iloc[list(pd.notnull(XAGE["Binned_age"]))]
yAGE_test = yAGE.iloc[list(pd.isnull(XAGE["Binned_age"]))]
XAGE_train.drop("Binned_age",axis=1,inplace=True)
XAGE_test.drop("Binned_age",axis=1,inplace=True)
print "shape(XAGE): {}, shape(XAGE_train): {}, shape(XAGE_test): {}".format(shape(XAGE),shape(XAGE_train),shape(XAGE_test))
print "len(yAGE): {}, len(yAGE_train): {}, len(yAGE_test): {}".format(len(yAGE),len(yAGE_train),len(yAGE_test))

#attempt to predict the missing age values using a RandomForestClassifier with
#gridsearch
X1,X2,y1,y2 = train_test_split(XAGE_train,yAGE_train)
#use RandomForestClassifier ?
param_grid={"n_estimators":[10,50,100,150,200]}
grid_search = GridSearchCV(RandomForestClassifier(),param_grid,cv=5)
grid_search.fit(X1,y1)
print "Age fitting complete... best params: {}".format(grid_search.best_params_)
print "Age train score: {}".format(grid_search.score(X1,y1)) 
print "Age test score: {}".format(grid_search.score(X2,y2)) 
#rebuild the model using the best params and all data before prediction:
grid_search=RandomForestClassifier(n_estimators=grid_search.best_params_["n_estimators"])
grid_search.fit(XAGE_train,yAGE_train)
print "Age train score: {}".format(grid_search.score(XAGE_train,yAGE_train)) 
yAGE_pred = grid_search.predict(XAGE_test) #predict the missing ages

for i in range(len(yAGE_pred)):
    yAGE_test.iloc[i] = yAGE_pred[i]

nanindex = TRAIN.Binned_age[pd.isnull(TRAIN.Binned_age)].index
#first settle the 177 NaN in TRAIN
for i in range(len(nanindex)):
    index_to_insert = nanindex[i]
    TRAIN.Binned_age.iloc[index_to_insert] = yAGE_test.iloc[i]
nanindex1 = TEST.Binned_age[pd.isnull(TEST.Binned_age)].index    
#next settle the 86 NaN in TEST
for i in range(len(nanindex1)):
    index_to_insert = nanindex1[i]
    TEST.Binned_age.iloc[index_to_insert] = yAGE_test.iloc[i+len(nanindex)]  

print "Binned age value_counts():"
print TRAIN.Binned_age.value_counts()
print TEST.Binned_age.value_counts()

#final null check
print "Number of null values per column in TRAIN:"
for i in TRAIN.columns:
    if i=="Age":
        print "Ignoring Age column in lieu of Binned_age..."
    else:
        if checkForNan(TRAIN,i) > 0:
            print "{}:{}".format(i,checkForNan(TRAIN,i))
print "Number of null values per column in TEST:"
for i in TEST.columns:
    if i=="Age":
        print "Ignoring Age column in lieu of Binned_age..."
    else:
        if checkForNan(TEST,i) > 0:
            print "{}:{}".format(i,checkForNan(TEST,i))

#end feature engineering

#extract features and target to train the machine with
features = ["Pclass","Sex","Binned_age","Is_alone","Companions","In_cabin","Binned_fare","Embarked","Honorific"]
X_train = TRAIN[features]
y_train = TRAIN["Survived"]
X_test = TEST[features]
#X1,X2,y1,y2 = train_test_split(X_train,y_train)

#if we use SVC we should scale the features first!
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)

#grid search using RandomForestClassifier
#param_grid={"n_estimators":[10,50,100,150,200]}
#rfc_grid = GridSearchCV(RandomForestClassifier(),param_grid,cv=5)
#rfc_grid.fit(X_train,y_train)
#print "Fitting complete... best params: {}".format(rfc_grid.best_params_)
#print "Train score: {}".format(rfc_grid.score(X_train,y_train)) 

#grid search using SVC
#param_grid={"C":[10**i for i in range(-5,5+1,1)],"gamma":[10**i for i in range(-5,5+1,1)]}
#svc_grid = GridSearchCV(SVC(kernel="rbf"),param_grid,cv=5)
#svc_grid.fit(X_train,y_train)
#print "Fitting complete... best params: {}".format(svc_grid.best_params_)
#print "Train score: {}".format(svc_grid.score(X_train,y_train)) 

#grid search using MLPClassifier
param_grid={"alpha":[10**i for i in range(-5,5+1,1)]}
mlpc_grid=GridSearchCV(MLPClassifier(),param_grid,cv=5)
mlpc_grid.fit(X_train,y_train)
print "Fitting complete... best params: {}".format(mlpc_grid.best_params_)
print "Train score: {}".format(mlpc_grid.score(X_train,y_train)) 

#make a prediction:
y_pred = mlpc_grid.predict(X_test)
y_pred = pd.DataFrame({"PassengerId":TEST.PassengerId,"Survived":y_pred})
y_pred.to_csv("survived.csv",index=False)

print "Elapsed time: {:.2f}s".format(time.time()-STARTTIME)

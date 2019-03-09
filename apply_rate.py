# -*- coding: utf-8 -*-
"""Apply_Rate.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RyVVT1F9WtI2PnKwhQgprOh6CSdjRVzm
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score, roc_auc_score



df = pd.read_csv('./Apply_Rate_2019.csv')
df.shape

df.columns

not_cat_cols = ['title_proximity_tfidf', 'description_proximity_tfidf', 'main_query_tfidf', 'query_jl_score', 'query_title_score',  'job_age_days']

df.head().T

df.describe()

# Number of unique entries
df.astype(object).describe(include='all').loc['unique', :]

# Number of Null Entries
df.isnull().sum(axis=0)

# Correlation Matrix
df.corr().style.background_gradient()

df.hist(figsize=(20,20))



print(df['apply'].value_counts())
print(df['apply'].value_counts()[0] / len(df) * 100)
print(df['apply'].value_counts()[1] / len(df) * 100)

df.fillna(0, inplace=True)
df['class_id'] = df['class_id'].astype('category').cat.codes

from sklearn.preprocessing import StandardScaler
# Preprocessing
def preprocess(to_remove, normalize=False):
    #Normalization
    df[not_cat_cols] = (df[not_cat_cols] - df[not_cat_cols].mean()) / (df[not_cat_cols].max() - df[not_cat_cols].min())
    
    df_train = df[df.search_date_pacific != '2018-01-27']
    df_test = df[df.search_date_pacific == '2018-01-27']
    
    global X_train
    X_train = df_train.drop(columns=to_remove).values
    global y_train 
    y_train  = df_train['apply'].values
    global X_test 
    X_test = df_test.drop(columns=to_remove).values
    global y_test 
    y_test = df_test['apply'].values
    
    #Normalization
#     if normalize:    
#         scaler = StandardScaler()
#         scaler.fit(X_train)
#         X_train = scaler.transform(X_train)
#         X_test = scaler.transform(X_test)

def classify(clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    print(clf)
    print("ROC AUC:", roc_auc_score(y_test, y_pred[:,1]))
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Detailed confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    
    
from imblearn.over_sampling import SMOTE

def classify_with_smote(clf):
    smote = SMOTE(random_state=0, n_jobs=-1, sampling_strategy=4/6)
    global X_train, y_train 
    X_train, y_train = smote.fit_resample(X_train, y_train)
    classify(clf)

to_remove = ['apply', 'search_date_pacific', 'class_id']
preprocess(to_remove)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0, class_weight='balanced', n_jobs=-1)
classify(clf)







from xgboost import XGBClassifier

clf = XGBClassifier(random_state=0, max_depth=8, n_jobs=-1, scale_pos_weight=6)
classify(clf)





to_remove = ['apply', 'search_date_pacific', 'class_id']
preprocess(to_remove, normalize=True)

clf = XGBClassifier(random_state=0, max_depth=8, n_jobs=-1)
classify(clf)

clf = XGBClassifier(random_state=0, max_depth=8, n_jobs=-1, scale_pos_weight=6)
classify(clf)



from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, class_weight='balanced', n_jobs=-1)
classify_with_smote(clf)

from xgboost import XGBClassifier

clf = XGBClassifier(random_state=0, max_depth=8, n_jobs=-1)
classify_with_smote(clf)





to_remove = ['apply', 'search_date_pacific']
preprocess(to_remove)

clf = XGBClassifier(random_state=0, max_depth=8, n_jobs=-1)
classify(clf)



clf = LogisticRegression(random_state=0, class_weight='balanced', n_jobs=-1)
classify(clf)



to_remove = ['apply', 'search_date_pacific']
preprocess(to_remove, normalize=True)

clf = XGBClassifier(random_state=0, max_depth=8, n_jobs=-1, reg_lambda=6)
classify(clf)



clf = LogisticRegression(C= 0.001, random_state=0, class_weight='balanced', n_jobs=-1)
classify(clf)



#Took forever to run

# from sklearn.svm import SVC

# clf = SVC(probability=True, C=0.01)
# classify(clf)

clf = XGBClassifier(random_state=0, max_depth=8, n_jobs=-1, reg_lambda=6)
classify_with_smote(clf)





clf = LogisticRegression(random_state=0, class_weight='balanced', n_jobs=-1)
classify_with_smote(clf)
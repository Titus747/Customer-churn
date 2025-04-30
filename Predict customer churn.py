"""import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
df=pd.read_csv("C:\\Users\\user\\Downloads\\WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df)
print(df.shape)
print(df.columns)
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.isnull().sum())
#df.drop(['Partner'],axis=1,inplace=True)
print(df)

Lb=LabelEncoder()
df["SeniorCitizen"]=Lb.fit_transform(df["SeniorCitizen"])
df["Contract"]=Lb.fit_transform(df["Contract"])
print(df)

x=df.drop(["Churn"],axis=1)
y=df[["Churn"]]
x_train, x_test, y_train, y_test = train_test_split(x, y,  random_state=42,test_size=0.3)
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)

model=SVC()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

acc=accuracy_score(y_test,y_pred)
print("Accurancy:",round(acc,2)*100,"%")

conf_matrix=confusion_matrix(y_test,y_pred)
print("Confusion matrix:",conf_matrix)

class_report=classification_report(y_test,y_pred)
print("classification report:",class_report)"""





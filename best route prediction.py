#!/usr/bin/env python
# coding: utf-8

# In[74]:


import pandas as pd


# In[75]:


import matplotlib.pyplot as plt


# In[76]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[77]:


import seaborn as sns
import numpy as np 
import math
import datetime as dt
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from statsmodels.tsa.api import Holt


# In[78]:


from sklearn.tree import DecisionTreeClassifier


# In[79]:


from sklearn.ensemble import RandomForestClassifier


# In[80]:


route = pd.read_csv("bestroute.csv")
route.head()


# In[81]:


print("size/shape of the datasrt", route.shape)
print("checking for null values", route.isnull().sum())
print("checking Data.Type", route.dtypes)


# In[82]:


sns.countplot(x="Taken", hue="transport(1,2)", data=route)


# In[83]:


df=sns.countplot("Taken", hue="safety(1,2,3)", data=route)


# In[84]:


sns.countplot(x="Taken", hue="traffic(0,1,2)", data=route)


# In[85]:


sns.countplot(x="Taken", hue="network avable(0,1,2)", data=route)


# In[86]:


sns.countplot(x="Taken", hue="diversions(0,1,2,3,4,5)", data=route)


# In[87]:


route["tot_dist"].plot.hist()


# In[88]:


route["safety(1,2,3)"].plot.hist()


# In[89]:


route.info(5)


# In[90]:


sns.boxplot(x="traffic(0,1,2)", y="safety(1,2,3)", data=route)


# In[91]:


route.isnull()


# In[92]:


route.isnull().sum()


# In[93]:


sns.heatmap(route.isnull(), yticklables=False, cmap="viridis")


# In[94]:


route.head(5)


# In[95]:


sns.heatmap(route.isnull(), yticklabels=False, cbar=False)


# In[96]:


route.drop("accident", axis=1, inplace=True)


# In[97]:


route.drop("diversions(0,1,2,3,4,5)", axis=1, inplace=True)


# In[98]:


route.head(5)


# In[99]:


sns.heatmap(route.isnull(), yticklabels=False, cbar=False)


# In[100]:


route.isnull().sum()


# In[101]:


route.drop(["tot_dist"], axis=1, inplace=True)
route.drop("via", axis=1, inplace=True)
route.drop("picpup", axis=1, inplace=True)
route.drop("drop", axis=1, inplace=True)


# Logestic Regression chk

# In[102]:


safety=pd.get_dummies(route['safety(1,2,3)'])


# In[103]:


safety.head(5)


# In[104]:


safety=pd.get_dummies(route['safety(1,2,3)'], drop_first=True)
safety.head(5)


# In[105]:


network=pd.get_dummies(route['network avable(0,1,2)'])
network.head(5)


# In[106]:


network=pd.get_dummies(route['network avable(0,1,2)'], drop_first=True)
network.head(5)


# In[107]:


road=pd.get_dummies(route['road_type(pakka, kaccha)'])
road.head(5)


# In[108]:


road=pd.get_dummies(route['road_type(pakka, kaccha)'], drop_first=True)
road.head(5)


# In[109]:


traffic=pd.get_dummies(route['traffic(0,1,2)'])
traffic.head(5)


# In[110]:


traffic=pd.get_dummies(route['traffic(0,1,2)'], drop_first=True)
traffic.head(5)


# In[111]:


route=pd.concat([route, traffic, road, network, safety], axis=1)


# In[112]:


route.head(5)


# Training

# In[114]:


x=route.drop("Taken", axis=1)
y=route["Taken"]


# In[115]:


from sklearn.model_selection import train_test_split


# In[116]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)


# In[117]:


from sklearn.linear_model import LogisticRegression


# In[118]:


logmodel=LogisticRegression(max_iter=1000)


# In[119]:


route.head(5)


# In[120]:


logmodel.fit(x_train, y_train)


# In[121]:


predictions=logmodel.predict(x_test)


# In[122]:


from sklearn.metrics import classification_report


# In[124]:


classification_report(y_test, predictions)


# In[125]:


from sklearn.metrics import confusion_matrix


# In[126]:


confusion_matrix(y_test, predictions)


# In[127]:


from sklearn.metrics import accuracy_score


# In[128]:


accuracy_score(y_test, predictions)


# In[129]:


decision_tree=DecisionTreeClassifier()


# In[130]:


decision_tree.fit(x_train, y_train)


# In[131]:


x_pred=decision_tree.predict(x_test)


# In[132]:


acc_decision_tree=round(decision_tree.score(x_train, y_train)*100, 2)


# In[133]:


acc_decision_tree


# In[134]:


route.pivot_table('Taken', index='transport(1,2)', columns='safety(1,2,3)').plot()


# In[135]:


route.pivot_table('Taken', index='network avable(0,1,2)', columns='safety(1,2,3)').plot()


# In[136]:


route.pivot_table('Taken', index='traffic(0,1,2)', columns='safety(1,2,3)').plot()


# Creat a function within many Machine learning Models

# In[137]:


def models(x_train, y_train):
       from sklearn.linear_model import LogisticRegression
       log = LogisticRegression(random_state=0)
       log.fit(x_train, y_train)
       
       from sklearn.tree import DecisionTreeClassifier
       tree=DecisionTreeClassifier(criterion='entropy', random_state=0)
       tree.fit(x_train, y_train)
       
       print('[0]Logistic Regression Training Accuracy:', log.score(x_train, y_train))
       print('[1]Decision Tree Classifier Training Accuracy:', tree.score(x_train, y_train))
       
       return log, tree


# In[139]:


model=models(x_train, y_train)


# In[147]:


picpup=(input("picpup location latitude:-"));
drop=(input("drop location latitude:-    "));
via=(input("via location latitude:-      "));

transport=int(input("1 for 2 wheel, 2 for 4 wheel:-"));
tot_dist=int(input("distance in km:- "));
safety=int(input("safety level measure in 1,2,3:-"));
network_avable=int(input("network availability rating (1,2,3):-"));
road_type=int(input("1 for pakka, 2 for kaccha:- "));
diversions=int(input("number of diversion:- "));
traffic=int(input("traffic measures level(1,2,3):- "));
accident=int(input("no. of accidents recorded:- "));
people_travelled= int(input("no. of people travelled:-"));


# In[148]:


road_taken=[[picpup, drop, via, transport, tot_dist, safety, network_avable, road_type, diversions, traffic, accident, people_travelled]]

pred=model[0].predict(road_taken)
print(pred)

if pred == 0:
    print("the road is taken")
else:
    print("the road is not taken")


# In[ ]:





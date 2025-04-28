#!/usr/bin/env python
# coding: utf-8

# # 1. Data cleaning and data visualization

# In[59]:


#import the libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[60]:


#Read the data with DataFrames
df = pd.read_csv('Data.csv')


# In[61]:


df


# In[62]:


# Summary of dataframe
df.info()


# In[78]:


df.shape


# In[63]:


#Remove the columns that are useless for our purpose, such as "id" and "Unnamed 32" columns.
df.drop("Unnamed: 32", axis=1, inplace=True)
df.drop('id',axis=1, inplace=True)


# In[64]:


df


# 

# In[65]:


# Count how many diagnosis are malignant (M) and how many are benignant (B)
plt.figure(figsize = (5,5))
sns.countplot(x="diagnosis", data=df, palette='crest')


# In[66]:


#Represent the DataFrame with a heatmap
plt.figure(figsize=(25,20))
sns.heatmap(df.corr(), annot=True,linewidths=.5, cmap="BuPu")


# # 2. Building ML models

# In[67]:


#Couting in the diagnosis column, the number of B and M.
df['diagnosis'].value_counts()


# In[68]:


#Convert categorical values (B,M) to binary values (0,1)
df['diagnosis']=df['diagnosis'].map({'B':0,'M':1})
df['diagnosis'].value_counts()


# ## 2.1 Split the data into tran and test

# In[69]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
                df.drop('diagnosis', axis=1),
                df['diagnosis'],
                test_size=0.2,
                random_state=42)

print("Shape of training set:", X_train.shape)
print("Shape of test set:", X_test.shape)


# In[70]:


#StandardScaler standardizes a feature by subtracting the mean and then scaling to unit variance.
#Unit variance means dividing all the values by the standard deviation.
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)


# ## 2.2 Logistic Regression Algorithm

# In[71]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
predictions1 = logreg.predict(X_test)


# In[101]:


from sklearn.metrics import accuracy_score

logreg_acc = accuracy_score(y_test, predictions1)
print("The accuracy of the Logistic Regression Algorithm is ", logreg_acc)


# ## 2.3 K-Nearest Neighbours Classifier Algorithm

# In[94]:


from sklearn.neighbors import KNeighborsClassifier


# In[95]:


# to find which value shows the lowest mean error
error_rate = []

for i in range(1,42):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[99]:


plt.figure(figsize=(10,5))
plt.plot(range(1,42), error_rate, color='red', linestyle="--",
         marker='o', markersize=10, markerfacecolor='b')
plt.title('Error_Rate vs K-value')
plt.show()


# K value of 9, 34, 35, 36, 40 and 41 show the lowest mean error.

# In[122]:


knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
predictions2 = knn.predict(X_test)


# In[123]:


knn_model_acc = accuracy_score(y_test, predictions2)
print("The accuracy of the K-Nearest Neighbors Classifier Algorithm is ", knn_model_acc)


# ## 2.3 Random Forests Classifier Algorithm

# In[133]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
predictions3 = rfc.predict(X_test)


# In[134]:


rfc_acc = accuracy_score(y_test, predictions3)
print("The accuracy of the Random Forests Classifier Algorithm is ", rfc_acc)


# ## 2.4 Support Vector Machines (SVM)
# 

# In[135]:


from sklearn.svm import SVC

svc = SVC(kernel="rbf")
svc.fit(X_train, y_train)
predictions4 = svc.predict(X_test)


# In[136]:


svm_acc = accuracy_score(y_test, predictions4)
print("The accuracy of SVM Algorithm is ", svm_acc)


# # 3. Conclusions

# The accuracy of the Logistic Regression Algorithm is 98.25%.
# The accuracy of the KNN Classifier Algorithm is 96.49%
# The accuracy of the Random Forest Classifier Algorithm is 96.49%
# The accuracy of the SVM Algorithm is 98.25%.
# 
# All in all, we have obtained a very accurated predictions.

# In[ ]:





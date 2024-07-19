#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


dataset = pd.read_csv("Train_2.csv")


# In[17]:


dataset.head()


# In[19]:


dataset.shape


# In[20]:


dataset.info()


# In[21]:


dataset.describe()


# In[23]:


pd.crosstab(dataset['Credit_History'],dataset['Loan_Status'], margins = True)


# In[26]:


dataset.boxplot(column='ApplicantIncome')


# In[27]:


dataset['ApplicantIncome'].hist(bins=20)


# In[28]:


dataset['CoapplicantIncome'].hist(bins=20)


# In[29]:


dataset.boxplot(column='ApplicantIncome', by= 'Education')


# In[30]:


dataset.boxplot(column = 'LoanAmount')


# In[33]:


dataset['LoanAmount'].hist(bins=20)


# In[34]:


dataset['LoanAmount_log'] = np.log(dataset['LoanAmount'])
dataset['LoanAmount_log'].hist(bins=20)


# In[36]:


dataset.isnull().sum()


# In[52]:


dataset['Gender'].fillna(dataset['Gender'].mode()[0], inplace = True)


# In[53]:


dataset['Married'].fillna(dataset['Married'].mode()[0], inplace = True)


# In[54]:


dataset['Dependents'].fillna(dataset['Dependents'].mode()[0], inplace = True)


# In[55]:


dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0], inplace = True)


# In[47]:


dataset.LoanAmount = dataset.LoanAmount.fillna(dataset.LoanAmount.mean())


# In[48]:


dataset.LoanAmount_log = dataset.LoanAmount_log.fillna(dataset.LoanAmount_log.mean())


# In[56]:


dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0], inplace = True)


# In[57]:


dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace = True)


# In[58]:


dataset.isnull().sum()


# In[59]:


dataset['TotalIncome']=dataset['ApplicantIncome']+dataset['CoapplicantIncome']
dataset['TotalIncome_log'] = np.log(dataset['TotalIncome'])


# In[60]:


dataset['TotalIncome_log'].hist(bins=20)


# In[61]:


dataset.head()


# In[63]:


x = dataset.iloc[:,np.r_[1:5,9:11,13:15]].values
y = dataset.iloc[:,12].values


# In[64]:


x


# In[65]:


y


# In[66]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state = 0)


# In[67]:


print(x_train)


# In[68]:


from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()


# In[69]:


for i in range (0,5):
    x_train[:,i]=labelencoder_X.fit_transform(x_train[:,i])


# In[70]:


x_train[:,7] = labelencoder_X.fit_transform(x_train[:,7])


# In[71]:


x_train


# In[73]:


labelencoder_Y = LabelEncoder()
y_train = labelencoder_Y.fit_transform(y_train)


# In[74]:


y_train


# In[75]:


for i in range (0,5):
    x_test[:,i]=labelencoder_X.fit_transform(x_test[:,i])


# In[76]:


x_test[:,7] = labelencoder_X.fit_transform(x_test[:,7])


# In[77]:


x_test


# In[78]:


labelencoder_Y = LabelEncoder()
y_test = labelencoder_Y.fit_transform(y_test)


# In[79]:


y_test


# In[80]:


from sklearn.preprocessing import StandardScaler
ss= StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)


# In[82]:


from sklearn.tree import DecisionTreeClassifier
DTClassifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
DTClassifier.fit(x_train,y_train)


# In[83]:


y_pred = DTClassifier.predict(x_test)


# In[84]:


y_pred


# In[86]:


from sklearn import metrics
print("The accuracy of thr decision tree is : ", metrics.accuracy_score(y_pred,y_test))


# In[87]:


from sklearn.naive_bayes import GaussianNB
NBClassifier = GaussianNB()
NBClassifier.fit(x_train,y_train)


# In[88]:


y_pred = NBClassifier.predict(x_test)


# In[89]:


y_pred


# In[90]:


print("The accuracy of thr Naive bayes is : ", metrics.accuracy_score(y_pred,y_test))


# In[92]:


testdata = pd.read_csv("train.csv")


# In[93]:


testdata.head()


# In[94]:


testdata.info()


# In[95]:


testdata.isnull().sum()


# In[97]:


testdata['Gender'].fillna(testdata['Gender'].mode()[0], inplace = True)
testdata['Dependents'].fillna(testdata['Dependents'].mode()[0], inplace = True)
testdata['Self_Employed'].fillna(testdata['Self_Employed'].mode()[0], inplace = True)
testdata['Credit_History'].fillna(testdata['Credit_History'].mode()[0], inplace = True)


# In[98]:


testdata['Loan_Amount_Term'].fillna(testdata['Loan_Amount_Term'].mode()[0], inplace = True)


# In[99]:


testdata.isnull().sum()


# In[101]:


testdata.boxplot(column='LoanAmount')


# In[103]:


testdata.boxplot(column='ApplicantIncome')


# In[104]:


testdata.LoanAmount = testdata.LoanAmount.fillna(testdata.LoanAmount.mean())


# In[106]:


testdata['LoanAmount_log'] = np.log(testdata['LoanAmount'])


# In[107]:


testdata.isnull().sum()


# In[108]:


testdata['TotalIncome'] = testdata['ApplicantIncome']+testdata['CoapplicantIncome']
testdata['TotalIncome_log']= np.log(testdata['TotalIncome'])


# In[109]:


testdata.head()


# In[111]:


test = testdata.iloc[:,np.r_[1:5,9:11,13:15]].values


# In[112]:


for i in range(0,5):
    test[:,i] = labelencoder_X.fit_transform(test[:,i])


# In[114]:


test[:,7] = labelencoder_X.fit_transform(test[:,7])


# In[115]:


test


# In[116]:


test = ss.fit_transform(test)


# In[117]:


pred = NBClassifier.predict(test)


# In[118]:


#Final output
pred


# In[ ]:




